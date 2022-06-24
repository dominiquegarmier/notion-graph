from __future__ import annotations

from typing import NoReturn

from flask import Flask
from flask import render_template
from flask import send_file

from api.config import config

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def data():
    return send_file(config.data_dir / 'data.json')


def server_main() -> int:
    app.run(host='0.0.0.0', port=8080)
    return 0
