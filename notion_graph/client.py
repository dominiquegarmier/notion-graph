from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from typing import cast

from aiohttp import ClientSession

from notion_graph.config import config


@asynccontextmanager
async def HeaderSession(auth: str | None = None):
    auth = auth or config.notion_key
    headers = {
        'Authorization': f'Bearer {auth}',
        'Notion-Version': '2022-02-22',
    }
    async with ClientSession(headers=headers) as session:
        yield session


async def _api_get(endpoint: str) -> dict[str, Any]:
    async with HeaderSession() as session:
        async with session.get(endpoint) as resp:
            return cast(dict[str, Any], await resp.json())


async def get_page(page_id: str) -> dict[str, Any]:
    return await _api_get(f'https://api.notion.com/v1/pages/{page_id}')


async def get_children(block_id: str) -> dict[str, Any]:
    return await _api_get(f'https://api.notion.com/v1/blocks/{block_id}/children')


async def get_block(block_id: str) -> dict[str, Any]:
    return await _api_get(f'https://api.notion.com/v1/blocks/{block_id}')
