from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import time
from functools import cached_property
from pathlib import Path
from queue import Queue
from typing import Any
from typing import cast
from typing import NamedTuple

from notion_client import AsyncClient

from notion_graph.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

NESTED_BLOCK_TYPES = [
    'paragraph',
    'bulleted_list_item',
    'numbered_list_item',
    'toggle',
    'to_do',
    'quote',
    'callout',
    'synced_block',
    'template',
    'column',
    'child_page',
    'child_database',
    'table',
]


class Page(NamedTuple):
    id: str
    url: str
    title: str


class Relation(NamedTuple):
    from_id: str
    to_id: str


class Block(NamedTuple):
    id: str
    page_id: str
    is_page: bool = False
    is_database: bool = False


class Parser:
    _queue: Queue[Block]
    _parsed_block_ids: set[str]
    _workers_done: list[bool]
    _queue_lock: asyncio.Lock

    pages: list[Page]
    relations: list[Relation]

    max_workers: int
    root_page: str
    queue_timeout: float

    def __init__(
        self, root_page: str, max_workers: int = 8, queue_timeout: float = 0.001
    ) -> None:
        self.max_workers = max_workers
        self.root_page = root_page
        self.queue_timeout = queue_timeout

        # mutable defaults
        self._workers_done = [False] * self.max_workers
        self._queue = Queue()
        self._queue_lock = asyncio.Lock()
        self._parsed_block_ids = set()
        self.pages = []
        self.relations = []

    async def parse(self) -> tuple[list[Page], list[Relation]]:
        try:
            await asyncio.wait_for(self._retry_parse(), timeout=config.max_parsing_time)
        except asyncio.TimeoutError:
            logger.error(
                f'parsing took too long, stopped after {config.max_parsing_time:.2f}s'
            )
        return self.pages, self.relations

    async def _retry_parse(self) -> None:
        while True:
            try:
                await self._parse()
            except Exception:
                pass
            else:
                break

    async def _parse(self) -> None:
        t_start = time.monotonic()
        # add initial block to queue
        self._queue.put(Block(id=self.root_page, is_page=True, page_id=self.root_page))

        workers = [
            asyncio.create_task(self.worker(worker_id=i))
            for i in range(self.max_workers)
        ]

        await asyncio.gather(*workers)
        t_end = time.monotonic()
        logger.info(f'parsing took {t_end - t_start} seconds')

    @cached_property
    def client(self) -> AsyncClient:
        return AsyncClient(auth=config.notion_key)

    @classmethod
    def find_key(cls, d: dict[str, Any], key: str) -> dict | None:
        for k, v in d.items():
            if k == key:
                return cast(dict, v)
            if isinstance(v, dict):
                found = cls.find_key(v, key)
                if found is not None:
                    return found
            if isinstance(v, list):
                for item in v:
                    found = cls.find_key(item, key)
                    if found is not None:
                        return found
        return None

    async def worker(self, worker_id: int) -> None:
        def _get_next_block_id() -> Block | None:
            try:
                block = self._queue.get(block=False)
            except queue.Empty:
                return None
            else:
                return block

        while True:
            try:
                timeout = asyncio.sleep(self.queue_timeout)  # type: ignore
                async with self._queue_lock:
                    block = _get_next_block_id()
                    if block is None:
                        self._workers_done[worker_id] = True
                        if all(self._workers_done):
                            break
                    else:
                        self._workers_done[worker_id] = False
                        self._parsed_block_ids.add(block.id)

                if block is not None:
                    await self.parse_block(block)
            finally:
                await timeout

    async def parse_block(self, block: Block) -> None:
        logger.debug(f'Parsing block {block!r}')
        if block.is_page:
            await self.parse_page(block.id)
        await self.parse_children(block_id=block.id, page_id=block.page_id)

    async def parse_page(self, page_id: str) -> None:
        page_dict = await self.client.pages.retrieve(
            page_id=page_id,
        )
        try:
            title = page_dict['properties']['title']['title'][0]['text']['content']
        except (KeyError, IndexError):
            title = f'Page: {page_dict["id"]}'

        if page_dict['icon'] and page_dict['icon']['type'] == 'emoji':
            title = f"{page_dict['icon']['emoji']} {title}"

        p = Page(id=page_id, url=page_dict['url'], title=title)
        logger.debug(p)
        self.pages.append(p)

    async def parse_children(self, block_id: str, page_id: str) -> None:
        children = (await self.client.blocks.children.list(block_id=block_id))[
            'results'
        ]
        for child in children:
            blocks: list[Block] = []

            if child['type'] == 'child_page':
                id = child['id']
                blocks.append(Block(id=id, is_page=True, page_id=id))

            if (v := self.find_key(child, 'mention')) is not None and v[
                'type'
            ] == 'page':
                id = v['page']['id']
                blocks.append(Block(id=id, is_page=True, page_id=id))
            if child['has_children']:
                blocks.append(Block(id=child['id'], page_id=page_id))

            for block in blocks:
                if block.is_page:
                    r = Relation(from_id=page_id, to_id=block.id)
                    logger.debug(r)
                    self.relations.append(r)

                async with self._queue_lock:
                    if block.id not in self._parsed_block_ids:
                        self._queue.put(block)


def page_to_dict(p: Page) -> dict[str, str]:
    return {'id': p.id, 'group': p.id, 'title': p.title, 'url': p.url}


def relation_to_dict(r: Relation) -> dict[str, str]:
    return {'source': r.from_id, 'target': r.to_id}


def write_json(pages: list[Page], relations: list[Relation]) -> None:
    obj = {
        'nodes': [page_to_dict(p) for p in pages],
        'links': [relation_to_dict(r) for r in relations],
    }

    config.data_dir.mkdir(parents=True, exist_ok=True)
    with open(config.data_dir / 'data.json', 'w') as f:
        f.write(json.dumps(obj, indent=2))
    logger.info('wrote pages and relatios to data.json')


async def amain() -> None:
    p = Parser(root_page='e601695a3c6f43ba842ce586924b0fdd')
    pages, relations = await p.parse()
    write_json(pages, relations)


def main() -> None:
    asyncio.run(amain())
