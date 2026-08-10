import argparse
import sys


def _process(args: argparse.Namespace) -> int:
    print("process is not implemented until milestone M2", file=sys.stderr)
    return 1


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("no_layups.main:app", host="127.0.0.1", port=8000)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="no-layups")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process", help="Run the pipeline on a video and write swing.json"
    )
    process_parser.add_argument("video", help="Path to the input video")
    process_parser.add_argument(
        "--handedness", choices=["right", "left"], default="right"
    )
    process_parser.add_argument(
        "-o", "--output", required=True, help="Path to write swing.json"
    )
    process_parser.set_defaults(func=_process)

    serve_parser = subparsers.add_parser("serve", help="Run the web server")
    serve_parser.set_defaults(func=_serve)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
