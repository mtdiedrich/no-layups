from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="No Layups")

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
