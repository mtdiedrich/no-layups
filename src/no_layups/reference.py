import json

from . import config


def load() -> dict:
    """Section 9.1: the bundled reference swing, in swing.json format."""
    return json.loads(config.REFERENCE_PATH.read_text())
