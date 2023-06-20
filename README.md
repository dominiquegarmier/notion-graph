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

## Development

### Installing

Clone this repo.

```
git clone git@github.com:dominiquegarmier/notion-graph
cd notion-graph
```

Install dependencies.

```
virtualenv .venv -ppython3.10
source .venv activate
pip install -r requirements.txt -r requirements-dev.txt
```

Install pre-commit hooks.

```
pre-commit install
```

### Setup

- Create a Notion key and store it in the `.env` file (see [`.temp.env`](.temp.env)).
- Share the relevant root file with your Notion key.
- Copy the uuid of your root file and paste it into the `.env` file.

### Usage

#### Parser

First you will need to parse your Notion Notes, this will generate a file `notion_graph/data/data.json` containing then graph nodes and edges for the frontend to plot in the second step. For this run the command:

```
python -m notion_graph parse
```

#### Frontend

Once you have generated the `data.json` file using the parser you can serve a simple html file with flask using the command

```
python -m notion_graph run
```

## Features / Improvments

### Parser

- [x] child pages
- [x] linked pages
- [ ] child databases (which are not linked elsewhere)
- [x] links to blocks
- [ ] blocks as nodes

### Frontend

- [x] display title
- [x] navigate to page on click
- [ ] hover for more information
- [ ] vertex and edge coloring (based on connection type, or something else?)
- [ ] display page icon without hover

### Technical

- [ ] better task queue
- [ ] tests
