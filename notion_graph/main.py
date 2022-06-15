from __future__ import annotations

from notion_graph.parser import main as parser_main
from notion_graph.server import app


def main() -> int:
    # parser_main()
    app.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
