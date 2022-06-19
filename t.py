from __future__ import annotations

import asyncio
from pprint import pprint

from notion_graph.client import get_page


async def amain():
    p = await get_page(page_id='e601695a3c6f43ba842ce586924b0fdd')
    pprint(p)


def main():
    asyncio.run(amain())


main()
