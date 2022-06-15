from __future__ import annotations

from notion_graph.parser import main as parser_main


def main() -> int:
    parser_main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
