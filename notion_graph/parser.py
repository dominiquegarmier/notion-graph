from __future__ import annotations

import asyncio
from collections import deque
from enum import Enum
from functools import cache
from typing import Any
from typing import cast
from typing import NamedTuple
from uuid import UUID
from uuid import uuid4

from notion_client import AsyncClient

from notion_graph.config import config


def find_key(d: dict[str, Any], key: str) -> Any:
    for k, v in d.items():
        if k == key:
            return v
        if isinstance(v, dict):
            found = find_key(v, key)
            if found is not None:
                return found
        if isinstance(v, list):
            for item in v:
                found = find_key(item, key)
                if found is not None:
                    return found
    return None


class Page(NamedTuple):
    id: str
    url: str
    title: str


class RelationType(Enum):
    CHILD_PAGE = 'child_page'
    MENTION = 'mention'


class Relation(NamedTuple):
    from_id: str
    to_id: str
    type: RelationType


async def parse_all(root: str) -> None:
    client = get_client()
    page_queue: deque[str] = deque([])
    parsed_pages: set[str] = set()
    page_queue.append(root)
    pages: list[Page] = []
    relations: list[Relation] = []

    page_queue.append(root)

    while page_queue:
        page_id = page_queue.pop()
        if page_id in parsed_pages:
            continue
        parsed_pages.add(page_id)
        print(page_id)

        resp = await client.pages.retrieve(page_id=page_id)
        pages.append(parse_page(resp))

        resp = await client.blocks.children.list(block_id=page_id)
        new_rels = await parser_blocks(resp['results'], page_id)
        for rel in new_rels:
            if rel.to_id not in parsed_pages:
                page_queue.append(rel.to_id)
        relations.extend(new_rels)

    from pprint import pprint

    pprint(pages)
    pprint(relations)


async def parser_blocks(blocks: list[dict[str, Any]], parent_id: str) -> list[Relation]:
    rels: list[Relation] = []
    for block in blocks:
        rels.extend(await parse_relation(block, parent_id))
    return rels


def parse_mention(block_dict: dict[str, Any]) -> str | None:
    mention = find_key(block_dict, 'mention')
    if mention is None:
        return None
    if mention['type'] == 'page':
        return cast(str, mention['page']['id'])
    return None


def parse_childpage(block_dict: dict[str, Any]) -> str | None:
    if block_dict['type'] == 'child_page':
        return cast(str, block_dict['id'])
    return None


async def parse_relation(block_dict: dict[str, Any], parent_id: str) -> list[Relation]:
    rels: list[Relation] = []

    rel = parse_childpage(block_dict)
    if rel is not None:
        rels.append(
            Relation(
                from_id=parent_id,
                to_id=rel,
                type=RelationType.CHILD_PAGE,
            )
        )
    rel = parse_mention(block_dict)
    if rel is not None:
        rels.append(
            Relation(
                from_id=parent_id,
                to_id=rel,
                type=RelationType.MENTION,
            )
        )

    if rels:
        return rels

    if 'has_children' in block_dict and block_dict['has_children']:
        resp = await get_client().blocks.children.list(block_id=block_dict['id'])
        rels.extend(await parser_blocks(resp['results'], parent_id))
    return rels


def parse_page(page_dict: dict[str, Any]) -> Page:
    try:
        title = page_dict['properties']['title']['title'][0]['text']['content']
    except (KeyError, IndexError):
        title = f'Page: {page_dict["id"]}'

    if page_dict['icon'] and page_dict['icon']['type'] == 'emoji':
        title = f"{page_dict['icon']['emoji']} {title}"

    return Page(id=page_dict['id'], url=page_dict['url'], title=title)


@cache
def get_client() -> AsyncClient:
    return AsyncClient(auth=config.notion_key)


async def amain() -> None:
    await parse_all('f57d968575854d1ea35f21c7ac01e3f7')


def main() -> None:
    asyncio.run(amain())
