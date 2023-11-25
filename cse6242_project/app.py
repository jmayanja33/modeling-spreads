from argparse import ArgumentParser

import uvicorn
from fastapi.middleware.wsgi import WSGIMiddleware

from cse6242_project.api.main import api_app
from cse6242_project.visuals.dashboard import dashapp


def get_user_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='port to run application on. Default=8000'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='port to run application on. Default=127.0.0.1'
    )
    args = parser.parse_args()
    return args


def run(host='127.0.0.1', port=8000):
    api_app.mount("/", WSGIMiddleware(dashapp.server))
    uvicorn.run(app=api_app, host=host, port=port)


def run_cli():
    args = get_user_args()
    run(args.host, args.port)
