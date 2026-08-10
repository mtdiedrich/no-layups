import argparse
import json
import sys
from pathlib import Path


def _process(args: argparse.Namespace) -> int:
    from no_layups.pipeline import PipelineError, run_pipeline

    def on_progress(step: str) -> None:
        print(f"[{step}]", file=sys.stderr)

    try:
        swing = run_pipeline(Path(args.video), args.handedness, on_progress)
    except PipelineError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        return 1

    Path(args.output).write_text(json.dumps(swing, indent=2))
    return 0


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
