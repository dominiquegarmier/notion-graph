from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import time
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from functools import wraps
from queue import Queue
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Generic
from typing import NamedTuple
from typing import ParamSpec
from typing import Protocol
from typing import TypeVar

from notion_client import AsyncClient

from notion_graph.config import config

logger = logging.getLogger(__name__)


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


class Task:
    def __init__(
        self,
        func: Callable[..., Awaitable[None]],
        _self: Parser,
        /,
        *args: Any,
        **kwargs: Any,
    ):
        self._func = func
        self._self = _self
        self.func_args = args
        self.func_kwargs = kwargs

    async def __call__(self) -> None:
        _func = self._func
        await _func(self._self, *self.func_args, **self.func_kwargs)
        return None

    def __repr__(self):
        return f'{self.__class__.__name__}(func={self._func}, _self={self._self})'


def task(func: Callable[..., Awaitable[None]]) -> Callable[..., None]:
    @wraps(func)
    def inner(_self: Parser, *args: Any, **kwargs: Any) -> None:
        task = Task(func, _self, *args, **kwargs)
        _self.task_queue.put_nowait(task)
        return None

    return inner


class TaskQueue(asyncio.Queue[Task]):
    done: bool = False

    def __init__(self, num_worker: int = 16):
        super().__init__()
        self.num_worker = num_worker

    async def __call__(self) -> None:
        async def _worker() -> None:
            while True:
                if self.done:
                    return None
                try:
                    task = await asyncio.wait_for(self.get(), timeout=0.1)
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
        self.done = True
        await asyncio.gather(*tasks)

    def push(self, task: Task) -> None:
        self.put_nowait(task)


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


class Graph:
    pages: set[Page]
    relations: set[Relation]


class Parser:
    task_queue = TaskQueue()
    graph: Graph
    parsed_blocks: set[str]

    root_id: str
    max_workers: int

    def __init__(self, root_id: str, num_workers: int = 16) -> None:

        self.parsed_blocks = set()
        self.graph = Graph()

        self.task_queue.num_worker = num_workers
        self.root_id = root_id

    def parse(self) -> Graph:
        self.parse_page(page_id=self.root_id)
        asyncio.run(self.task_queue())
        return self.graph

    @task
    async def parse_page(self, page_id: str) -> None:
        raise NotImplementedError()

    @task
    async def parse_database(self, database_id: str) -> None:
        raise NotImplementedError()

    @task
    async def parse_block(self, block_id: str) -> None:
        raise NotImplementedError()

    @task
    async def parse_children(self, block_id: str = '') -> None:
        raise NotImplementedError()

    @task
    async def parse_block_relation(self, block_id: str) -> None:
        raise NotImplementedError()


def parser_main() -> int:
    parser = Parser(root_id=config.root_id)
    graph = parser.parse()
    print(graph)
    return 0
