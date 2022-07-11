from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Awaitable
from collections.abc import Callable
from functools import wraps
from pprint import pprint
from typing import Any
from typing import cast
from typing import Concatenate
from typing import Generic
from typing import NamedTuple
from typing import ParamSpec
from uuid import UUID

from notion_graph.client import get_block
from notion_graph.client import get_children
from notion_graph.client import get_page
from notion_graph.config import config

logger = logging.getLogger(__name__)


P = ParamSpec('P')


class Task(Generic[P]):
    def __init__(
        self,
        func: Callable[Concatenate[Parser, P], Awaitable[None]],
        _self: Parser,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        self._func = func
        self._self = _self
        self.func_args = args
        self.func_kwargs = kwargs

    async def __call__(self) -> None:
        await self._func(self._self, *self.func_args, **self.func_kwargs)
        return None

    def __repr__(self):
        return f'{self.__class__.__name__}(func={self._func}, _self={self._self})'


def task(
    func: Callable[Concatenate[Parser, P], Awaitable[None]]
) -> Callable[Concatenate[Parser, P], None]:
    @wraps(func)
    def inner(_self: Parser, *args: Any, **kwargs: Any) -> None:
        task = Task(func, _self, *args, **kwargs)
        _self.task_queue.put_nowait(task)
        return None

    return inner


class TaskQueue(asyncio.Queue[Task]):
    done: asyncio.Event

    def __init__(self, num_worker: int = 64):
        super().__init__()
        self.num_worker = num_worker
        self.done = asyncio.Event()

    async def __call__(self) -> None:
        signal = object()

        async def _sig() -> object:
            await self.done.wait()
            return signal

        async def _worker() -> None:
            while True:
                task: None | Task = None
                try:
                    ts, pending = await asyncio.wait(
                        [self.get(), _sig()], return_when=asyncio.FIRST_COMPLETED
                    )
                    for p in pending:
                        p.cancel()
                    t = ts.pop().result()
                    if t is signal:
                        return None
                    else:
                        task = cast(Task, t)

                except asyncio.TimeoutError:
                    continue
                try:
                    await task()
                except Exception:
                    logger.warning(f'task failed: {task}')
                    raise  # should we raise or not?
                finally:
                    self.task_done()

        tasks = [asyncio.create_task(_worker()) for _ in range(self.num_worker)]
        await self.join()
        self.done.set()
        await asyncio.gather(*tasks)

    def push(self, task: Task) -> None:
        self.put_nowait(task)


class Page(NamedTuple):
    id: UUID
    url: str
    title: str


class Relation(NamedTuple):
    from_page: UUID
    to_page: UUID
    from_block: UUID
    to_block: UUID


class Graph:
    pages: set[Page]
    relations: set[Relation]

    def __init__(self) -> None:
        self.pages = set()
        self.relations = set()

    def _to_json(self) -> str:
        obj: dict[str, list[dict]] = {'nodes': [], 'links': []}
        for page in self.pages:
            obj['nodes'].append(
                {
                    'id': str(page.id),
                    'url': page.url,
                    'title': page.title,
                    'group': str(page.id),
                }
            )

        for relation in self.relations:
            obj['links'].append(
                {
                    'source': str(relation.from_page),
                    'target': str(relation.to_page),
                }
            )
        return json.dumps(obj, indent=2)

    def dump_json(self) -> None:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        with open(config.data_dir / 'data.json', 'w') as f:
            f.write(self._to_json())


class Parser:
    task_queue = TaskQueue()
    graph: Graph
    parsed_blocks: set[str]
    parsed_blocks_lock = asyncio.Lock()

    root_id: str
    max_workers: int

    def __init__(self, root_id: str, num_workers: int = 16) -> None:

        self.parsed_blocks = set()
        self.graph = Graph()

        self.task_queue.num_worker = num_workers
        self.root_id = root_id

    def parse(self) -> Graph:
        t_start = time.monotonic()

        self.parse_page(page_id=self.root_id)
        asyncio.run(self.task_queue())

        t_end = time.monotonic()
        logger.info(f'parsing took {t_end - t_start:.2f} seconds')
        return self.graph

    @task
    async def parse_page(self, page_id: str) -> None:
        def _parse_title(page_dict: dict[str, Any]) -> str:
            try:
                icon = cast(str, page_dict['icon']['emoji'])
            except KeyError:
                icon = None

            title: str | None = None

            props = page_dict['properties']
            title_prop = props.get(
                'title', props.get('Name')
            )  # depending on if parent is a database the title might be called Name
            assert title_prop is not None  # there should always be a title

            # construct title from richtext blocks
            title = ''
            for text in title_prop['title']:
                if text['type'] == 'text':
                    title += cast(str, text['text']['content'])
                else:
                    title += '-'

            if title is None:
                title = cast(str, page_dict['id'])
            if icon is not None:
                title = f'{icon} {title}'
            return title

        page_dict = await get_page(page_id=page_id)
        page = Page(
            id=UUID(page_dict['id']),
            url=page_dict['url'],
            title=_parse_title(page_dict),
        )
        logger.debug(f'added new page: {page}')
        self.graph.pages.add(page)
        self.parse_children(block_id=page_id, page_id=page_id)
        return

    @task
    async def parse_database(self, database_id: str) -> None:
        return

    @task
    async def parse_children(self, block_id: str, page_id: str) -> None:
        children = (await get_children(block_id=block_id))['results']
        for block_dict in children:
            await self._parse_block(
                block_dict=block_dict, block_id=block_dict['id'], page_id=page_id
            )

    async def _parse_block(self, block_dict: dict, block_id: str, page_id: str) -> None:
        async with self.parsed_blocks_lock:
            if block_id in self.parsed_blocks:
                return
            else:
                self.parsed_blocks.add(block_id)

        def _get_relations(
            block_dict: dict, block_id: str, page_id: str
        ) -> list[Relation]:
            ret: list[Relation] = []
            block_type = block_dict['type']
            rt = block_dict[block_type].get('rich_text', [])
            for text in rt:
                dest_page = dest_block = None
                if text['type'] == 'mention':
                    try:
                        dest_page = text['mention']['page']['id']
                    except KeyError:  # incase link to database
                        pass
                elif 'href' in text and text['href'] is not None:
                    href: str = text['href']
                    if href.startswith('/'):
                        href = href[1:]
                        if '#' in href:
                            dest_page, dest_block = href.split('#')
                        else:
                            dest_page = href

                if dest_page is not None:
                    rel = Relation(
                        from_page=UUID(page_id),
                        to_page=UUID(dest_page),
                        from_block=UUID(block_id),
                        to_block=UUID(dest_block or dest_page),
                    )
                    ret.append(rel)
            return ret

        if block_dict['type'] == 'child_page':
            relation = Relation(
                from_page=UUID(page_id),
                to_page=UUID(block_dict['id']),
                from_block=UUID(block_id),
                to_block=UUID(block_dict['id']),
            )
            self.graph.relations.add(relation)
            self.parse_page(page_id=block_dict['id'])
            return

        elif block_dict['type'] == 'child_database':
            # logger.warning('child_database not implemented')
            return

        else:
            if block_dict['has_children']:
                self.parse_children(block_id=block_id, page_id=page_id)

            relations = _get_relations(block_dict, block_id, page_id)
            for relation in relations:
                self.graph.relations.add(relation)
                self.parse_page(page_id=str(relation.to_page))


def parser_main() -> int:
    parser = Parser(root_id=config.root_id)
    graph = parser.parse()
    graph.dump_json()
    return 0
