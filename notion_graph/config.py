from __future__ import annotations

import os
from functools import cache
from typing import NamedTuple

from dotenv import load_dotenv

__all__ = [
    'Config',
    'config',
    'get_config',
]


class Config(NamedTuple):
    notion_key: str


@cache
def get_config() -> Config:
    load_dotenv()
    return Config(notion_key=os.environ['NOTION_KEY'])


config = get_config()
