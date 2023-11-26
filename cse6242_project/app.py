from argparse import ArgumentParser

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
    dashapp.run(debug=True, host=host, port=port)

def run_cli():
    args = get_user_args()
    run(args.host, args.port)
