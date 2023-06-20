from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import AsyncGenerator
from collections.abc import Collection
from collections.abc import Generator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from logging import getLogger
from pathlib import Path
from threading import Thread
from typing import Any
from typing import Literal
from typing import NoReturn
from uuid import UUID
from uuid import uuid4

from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from dateutil.parser import parse as parse_date
from dotenv import load_dotenv
from flask import Flask
from flask import jsonify
from flask import render_template
from werkzeug import Response

logger = getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

NOTION_URL = 'https://www.notion.so'
NOTION_API_URL = 'https://api.notion.com/v1'

DEFAULT_DATA_PATH = Path(__file__).parent / 'data'
DEFAULT_REFRESH_INTERVAL = 60 * 5
DEFAULT_N_WORKERS = 4

# api config
RATE_LIMIT_BURST = 1
RATE_LIMIT = 3 * RATE_LIMIT_BURST
RATE_LIMITER = AsyncLimiter(RATE_LIMIT, RATE_LIMIT_BURST)
TIMEOUT = 30
MAX_RETRY = 1

SKIP_PROPAGATION_BLOCK_TYPES = (
    'child_page',
    'child_database',
)


@dataclass(frozen=True)
class Config:
    notion_key: str
    data_path: Path
    refresh_interval: int
    n_workers: int


def load_config() -> Config:
    load_dotenv()
    try:
        notion_key = os.environ['NOTION_KEY']
    except KeyError:
        raise ValueError('Missing NOTION_KEY environment variable')

    data_path = Path(os.environ.get('GRAPH_DATA_PATH', DEFAULT_DATA_PATH))
    refresh_interval = int(
        os.environ.get('GRAPH_REFRESH_INTERVAL', DEFAULT_REFRESH_INTERVAL)
    )
    n_workers = int(os.environ.get('GRAPH_N_WORKERS', DEFAULT_N_WORKERS))

    return Config(notion_key, data_path, refresh_interval, n_workers)


@dataclass
class Page:
    id: UUID
    url: str
    title: str
    last_parsed: datetime

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Page):
            return NotImplemented
        return self.id == other.id


def serialize_page(page: Page) -> dict[str, str]:
    return {
        'id': str(page.id),
        'url': str(page.url),
        'title': str(page.title),
        'last_parsed': page.last_parsed.isoformat(),
    }


def deserialize_page(data: dict[str, str]) -> Page:
    return Page(
        id=UUID(data['id']),
        url=data['url'],
        title=data['title'],
        last_parsed=datetime.fromisoformat(data['last_parsed']),
    )


@dataclass
class Link:
    id: UUID
    source: UUID
    target: UUID
    link_type: Literal['child', 'mention', 'href']

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Link):
            return NotImplemented
        return self.id == other.id


def serialize_link(link: Link) -> dict[str, str]:
    return {
        'id': str(link.id),
        'source': str(link.source),
        'target': str(link.target),
        'link_type': link.link_type,
    }


def deserialize_link(data: dict[str, str]) -> Link:
    return Link(
        id=UUID(data['id']),
        source=UUID(data['source']),
        target=UUID(data['target']),
        link_type=data['link_type'],  # type: ignore
    )


class Graph:
    _pages: dict[UUID, Page]
    _links: dict[UUID, Link]

    def __init__(
        self,
        pages: Collection[Page] | None = None,
        links: Collection[Link] | None = None,
    ) -> None:
        self._pages = {}
        self._links = {}

        if pages is None:
            pages = []
        if links is None:
            links = []

        for page in pages:
            self.add(page)
        for link in links:
            self.add(link)

    def __contains__(self, item: Page | UUID) -> bool:
        if isinstance(item, Page):
            return item.id in self._pages
        elif isinstance(item, UUID):
            return item in self._pages
        else:
            raise TypeError(f'Cannot check for item of type {type(item)}')

    @classmethod
    def deserialize(cls, data: str) -> Graph:
        dct = json.loads(data)
        try:
            pages = [deserialize_page(page) for page in dct['pages']]
            links = [deserialize_link(link) for link in dct['links']]
        except Exception as e:
            raise ValueError(f'Invalid data {e}')
        return cls(pages, links)

    def serialize(self) -> str:
        pages = [serialize_page(page) for page in self._pages.values()]
        links = [serialize_link(link) for link in self._links.values()]
        return json.dumps({'pages': pages, 'links': links})

    def save(self, path: str | os.PathLike) -> None:
        with open(path, 'w') as f:
            f.write(self.serialize())

    @classmethod
    def load(cls, path: str | os.PathLike) -> Graph:
        with open(path) as f:
            return cls.deserialize(f.read())

    @property
    def pages(self) -> list[Page]:
        return list(self._pages.values())

    @property
    def links(self) -> list[Link]:
        return list(self._links.values())

    def add(self, item: Page | Link) -> None:
        if isinstance(item, Page):
            if item.id in self._pages:
                raise ValueError(f'Page {item.id} already exists')
            self._pages[item.id] = item
        elif isinstance(item, Link):
            if item.id in self._links:
                raise ValueError(f'Link {item.id} already exists')
            self._links[item.id] = item
        else:
            raise TypeError(f'Cannot add item of type {type(item)}')

    def prune(self) -> Graph:
        links = set()
        for link in self._links.values():
            if link.source in self and link.target in self:
                links.add(link)
        return self.__class__(list(self._pages.values()), links)

    def update(self, pages: Collection[Page], links: Collection[Link]) -> None:
        new_pages = {page.id: page for page in pages}

        delete_ids = []
        for link in self._links.values():
            if link.source in new_pages:
                delete_ids.append(link.id)
        for id_ in delete_ids:
            del self._links[id_]

        self._pages.update(new_pages)
        self._links.update({link.id: link for link in links})


@dataclass
class DisplayLink:
    source: str
    target: str
    rotation: float
    curvature: float


@dataclass
class DisplayNode:
    id: str
    title: str
    url: str


@dataclass
class DisplayGraph:
    nodes: list[DisplayNode]
    links: list[DisplayLink]


def to_display_graph(graph: Graph) -> DisplayGraph:
    node_ids: set[str] = set()
    nodes = []
    for page in graph.pages:
        node_ids.add(str(page.id))
        node = DisplayNode(id=str(page.id), title=page.title, url=page.url)
        nodes.append(node)

    links_dict: dict[tuple[UUID, UUID], list[Link]] = defaultdict(list)
    for link in graph.links:
        if str(link.source) not in node_ids or str(link.target) not in node_ids:
            continue
        links_dict[(link.source, link.target)].append(link)

    links_list = []
    for ids_tp, links in links_dict.items():
        if ids_tp[0] == ids_tp[1]:
            base_curvature = 0.5
        else:
            base_curvature = 0
        n = len(links)
        for i, link in enumerate(links):
            rotation = 2 * math.pi * i / n
            new_link = DisplayLink(
                source=str(link.source),
                target=str(link.target),
                rotation=rotation,
                curvature=base_curvature + min((n - 1) / 10, 0.5),
            )
            links_list.append(new_link)

    return DisplayGraph(nodes=nodes, links=links_list)


@contextmanager
def persisted_graph(
    path: str | Path, flush: bool = False, persist: bool = True
) -> Generator[Graph, None, None]:
    if not flush:
        try:
            graph = Graph.load(path)
        except Exception:
            graph = Graph()
            logger.warning(f'Could not load graph from {path}, creating new graph.')
    else:
        graph = Graph()

    try:
        yield graph
    except Exception:
        raise
    else:
        if persist:
            fd, tmp_path = tempfile.mkstemp()
            with open(fd, 'w') as f:
                f.write(graph.serialize())
            os.replace(tmp_path, path)


@asynccontextmanager
async def RateLimitedSession(
    config: Config,
    auth: str | None = None,
) -> AsyncGenerator[ClientSession, None]:
    auth = auth or config.notion_key
    headers = {
        'Authorization': f'Bearer {auth}',
        'Notion-Version': '2022-02-22',
    }
    async with RATE_LIMITER:
        async with ClientSession(headers=headers) as session:
            yield session


async def paginated(
    method: Literal['GET', 'POST'],
    url: str,
    config: Config,
    initial_params: dict[str, Any] | None = None,
) -> list[dict]:
    results = []

    cursor: str | None = None
    has_more = True

    while has_more:
        params = initial_params or {}
        if cursor is not None:
            params = params | {'start_cursor': cursor}
        data = {}
        async with RateLimitedSession(config=config) as session:
            if method == 'GET':
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
            elif method == 'POST':
                async with session.post(url, json=params) as resp:
                    data = await resp.json()

        results.extend(data['results'])

        cursor = data.get('next_cursor')
        has_more = data.get('has_more', False) and cursor is not None

    return results


def _uuid_to_url(uuid: UUID) -> str:
    return f"{NOTION_URL}/{str(uuid).replace('-', '')}"


def _strip_uuid(href: str) -> UUID:
    if not href.startswith('/'):
        raise ValueError
    no_slash = href[1:]
    try:
        return UUID(no_slash.split('#')[0])
    except ValueError:
        pass
    try:
        return UUID(no_slash.split('?')[0])
    except ValueError:
        raise


def _parse_page(
    page_data: dict[str, Any], last_parsed: dict[UUID, datetime]
) -> Page | None:
    # skip archived pages
    if page_data['archived']:
        return None

    # only parse page if it has been updated since last parse
    page_id = UUID(page_data['id'])
    if page_id in last_parsed:
        last_edited = parse_date(page_data['last_edited_time'])

        time = last_parsed[page_id]
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        if last_edited < last_parsed[page_id]:
            return None

    properties = page_data.get('properties', {})
    for value in properties.values():
        if isinstance(value, dict) and value.get('type') == 'title':
            title_rich_text = value.get('title', [])
            break
    else:
        title_rich_text = []

    title_rich_text = [rt for rt in title_rich_text if rt['type'] == 'text']
    title = '-'.join([rt['text']['content'] for rt in title_rich_text])

    return Page(
        id=page_id,
        url=_uuid_to_url(page_id),
        title=title,
        last_parsed=datetime.now(timezone.utc),
    )


async def parse_pages(last_parsed: dict[UUID, datetime], config: Config) -> set[Page]:
    param = {'filter': {'value': 'page', 'property': 'object'}}
    logger.debug('getting page ids...')
    resp = await paginated(
        'POST', url=f'{NOTION_API_URL}/search', config=config, initial_params=param
    )

    ret = []
    for data in resp:
        page = _parse_page(page_data=data, last_parsed=last_parsed)
        if page is None:
            continue
        ret.append(page)
    return set(ret)


def _parse_rich_text(page: UUID, rich_text: dict[str, Any]) -> Link | None:
    if rich_text['type'] == 'mention':
        mention = rich_text['mention']
        if mention['type'] == 'page':
            return Link(
                id=uuid4(),
                source=page,
                target=UUID(mention['page']['id']),
                link_type='mention',
            )
    elif rich_text['type'] == 'text':
        if rich_text.get('href') is not None:
            try:
                uuid = _strip_uuid(rich_text['href'])
            except ValueError:
                logger.debug(f"failed to parse href format: {rich_text['href']}")
                return None
            return Link(id=uuid4(), source=page, target=uuid, link_type='href')
    return None


def parse_links(page: UUID, data: dict[str, Any]) -> list[Link]:
    block_type = data['type']
    if block_type in ('child_page', 'child_database'):
        return []
    if block_type not in data:
        return []

    ret: list[Link] = []
    block = data.get(block_type, {})
    for rich_text in block.get('rich_text', []):
        try:
            link = _parse_rich_text(page=page, rich_text=rich_text)
        except KeyError:
            pass
        else:
            if link is not None:
                ret.append(link)
    return ret


async def parse_children(
    page: UUID, block: UUID, config: Config
) -> tuple[list[UUID], list[Link]]:
    # logger.info(f"parsing children of {block} in {page}...")
    resp = await paginated(
        'GET', url=f'{NOTION_API_URL}/blocks/{block}/children', config=config
    )

    links: list[Link] = []
    children: list[UUID] = []

    for data in resp:

        # handle child_pages separately
        if data['type'] in 'child_page':
            links.append(
                Link(id=uuid4(), source=page, target=data['id'], link_type='child')
            )

        # handle any other links such as mentions and hrefs
        links.extend(parse_links(page=page, data=data))

        # handle propagation to children
        if data['type'] in SKIP_PROPAGATION_BLOCK_TYPES:
            continue

        if data.get('has_children'):
            children.append(UUID(data['id']))

    return children, links


@dataclass
class Task:
    page: UUID
    block: UUID
    retry: int = 0

    def __hash__(self) -> int:
        return hash(self.block)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self.block == other.block


async def worker(
    queue: asyncio.Queue[Task],
    links: set[Link],
    enqueued: set[Task],
    done: set[Task],
    failed: set[Task],
    config: Config,
):
    while True:
        task = await queue.get()
        try:
            children, new_links = await asyncio.wait_for(
                parse_children(page=task.page, block=task.block, config=config), TIMEOUT
            )
        except Exception:
            if task.retry >= MAX_RETRY:
                logger.debug(f'task failed: {task!r}')
                failed.add(task)
            else:
                logger.debug(f'retrying task: {task!r}')
                task.retry += 1
                queue.put_nowait(task)
        else:
            async with asyncio.Lock():
                for child in children:
                    # don't parse blocks twice
                    if child not in enqueued:
                        new_task = Task(page=task.page, block=child)
                        enqueued.add(new_task)
                        queue.put_nowait(new_task)

            for link in new_links:
                links.add(link)
            done.add(task)
        finally:
            queue.task_done()


async def parse(
    last_parsed: dict[UUID, datetime],
    config: Config,
) -> tuple[set[Page], set[Link]]:
    pages = await parse_pages(config=config, last_parsed=last_parsed)
    links: set[Link] = set()

    # monitor the queue
    queue: asyncio.Queue[Task] = asyncio.Queue()
    enqueued: set[Task] = set()
    done: set[Task] = set()
    failed: set[Task] = set()

    workers = []
    for _ in range(config.n_workers):
        task = asyncio.create_task(
            worker(queue, links, enqueued, done, failed, config=config)
        )
        workers.append(task)

    for page in pages:
        new_task = Task(page=page.id, block=page.id)
        queue.put_nowait(new_task)
        enqueued.add(new_task)

    # wait for all tasks to be done
    async def monitor() -> None:
        while True:
            await asyncio.sleep(1)
            logger.debug(
                f'ENQUEUED: {len(enqueued)}, DONE: {len(done)}, FAILED: {len(failed)}'
            )

    logger_task = asyncio.create_task(monitor())

    await queue.join()

    logger_task.cancel()
    logger.debug('work done, cancelling workers...')

    for w in workers:
        w.cancel()

    logger.info(f'done: {len(done)}, failed: {len(failed)}')

    return pages, links


async def partial_parse(config: Config, flush: bool = False) -> None:
    with persisted_graph(config.data_path / 'graph.json', flush=flush) as graph:
        last_parsed = {page.id: page.last_parsed for page in graph.pages}
        pages, links = await parse(last_parsed=last_parsed, config=config)
        graph.update(pages, links)


async def run_daemon(config: Config) -> NoReturn:
    while True:
        try:
            logger.info('refreshing graph...')
            await partial_parse(config=config, flush=False)
            await asyncio.sleep(config.refresh_interval)
        except Exception:
            logger.exception('error while parsing, retrying in 5s...')
            await asyncio.sleep(5)


def flask_app(config: Config) -> Flask:
    app = Flask(__name__)

    def index() -> Any:
        return render_template('index.html')

    def data() -> Response:
        with persisted_graph(config.data_path / 'graph.json') as graph:
            display_graph = to_display_graph(graph)
            return jsonify(dataclasses.asdict(display_graph))

    app.add_url_rule('/', view_func=index)
    app.add_url_rule('/data', view_func=data)
    return app


def main() -> int:
    config = load_config()
    app = flask_app(config)

    daemon = Thread(target=lambda: asyncio.run(run_daemon(config)))
    flask = Thread(target=lambda: app.run(host='0.0.0.0', port=8080))

    daemon.start()
    flask.start()

    daemon.join()
    flask.join()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
