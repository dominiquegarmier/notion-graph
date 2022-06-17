from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

__all__ = [
    'Config',
    'config',
    'get_config',
]


class Config(NamedTuple):
    notion_key: str
    root_id: str
    max_parsing_time: int = 60
    data_dir: Path = Path(__file__).parent / 'data'


@cache
def get_config() -> Config:
    load_dotenv()
    notion_key = os.environ['NOTION_KEY']
    root_id = os.environ['ROOT_ID']
    return Config(notion_key=notion_key, root_id=root_id)


config = get_config()
