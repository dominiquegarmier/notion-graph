from __future__ import annotations

import asyncio
from pprint import pprint

from notion_graph.client import get_block
from notion_graph.client import get_children
from notion_graph.client import get_page


async def amain():
    p = await get_children(block_id='453776be35854f74b098f7f6529ae33a')
    pprint(p)


def main():
    asyncio.run(amain())


main()
