from __future__ import annotations

import argparse
import logging
from typing import cast

from api.parser import parser_main
from api.server import server_main


def main() -> int:

    logging.basicConfig(level=logging.INFO, format='%(message)s')

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
