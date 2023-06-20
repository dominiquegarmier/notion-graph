[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/DominiqueGarmier/notion-graph/main.svg)](https://results.pre-commit.ci/latest/github/DominiqueGarmier/notion-graph/main)

# notion-graph

opensource graph view of your notion pages, inspired by [Obsidian](https://obsidian.md/).

## WARNING THIS IS STILL IN DEVELOPMENT

#### what currently works:

- a simple flask server (see the gif below)
- background parsing and auto updating (parses every X minutes automatically)
- retrying logic (it hasn't crashed into an unrecoverable state for me yet)
- partial updates (only parse pages that were edited since last parse)

<p align="center">
  <img src="https://github.com/DominiqueGarmier/notion-graph/assets/42445422/9735496a-fdd7-4ba0-a8df-7acacbba3f28" alt="notion-graph preview"/>
</p>

## Installing

Clone this repo.

```
git clone git@github.com:dominiquegarmier/notion-graph
cd notion-graph
```

Install dependencies.

```
virtualenv .venv -ppython3.10
source .venv activate
pip install -r requirements.txt
```

## Development

Install dev-dependencies

```
pip install -r requirements-dev.txt
```

Install pre-commit hooks.

```
pre-commit install
```

## Setup

- set the environment variable `NOTION_KEY` with your notion api key that has read access to some pages (see [notion docs]("https://developers.notion.com/docs/create-a-notion-integration")).

## Usage

you can now run the following command to start notion-graph

```
python graph.py
```

This will automatically discover any page shared with your notion integration. Subsequently it will create a task queue to query every discovered page. The initial parse of your document might take a while as notions api is limited to three requests per second. You will notice that the programm will create a new folder `data/` which contains the parsed pages and links. Subsequent parses will only refresh pages that have be edited since the last parse.

The graph view will be served on `localhost:8080`. Make sure to hit refresh when the parsing is done.
