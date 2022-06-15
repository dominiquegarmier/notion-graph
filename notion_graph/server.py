from __future__ import annotations

from flask import Flask
from flask import render_template
from flask import send_file

from notion_graph.config import config

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def data():
    return send_file(config.data_dir / 'data.json')
