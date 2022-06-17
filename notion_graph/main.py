from __future__ import annotations

import argparse
from typing import cast

from notion_graph.parser import parser_main
from notion_graph.server import server_main


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    server_parser = subparsers.add_parser('run')
    server_parser.set_defaults(func=server_main)

    parser_parser = subparsers.add_parser('parse')
    parser_parser.set_defaults(func=parser_main)

    args = parser.parse_args()
    return cast(int, args.func())


if __name__ == '__main__':
    raise SystemExit(main())
