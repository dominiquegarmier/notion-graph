from __future__ import annotations

import asyncio
from pprint import pprint

from notion_graph.client import get_block
from notion_graph.client import get_children
from notion_graph.client import get_page


async def amain():
    p = await get_children(block_id='2465521b79364efba808e29cde228616')
    pprint(p)


def main():
    asyncio.run(amain())


main()
