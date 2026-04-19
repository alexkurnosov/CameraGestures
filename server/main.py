"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from routers import admin, auth, examples, model, training


def _read_version() -> str:
    try:
        return (Path(__file__).parent / "VERSION").read_text().strip()
    except OSError:
        return "0.0.0"


SERVER_VERSION = _read_version()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure data directories and DB tables exist
    settings.ensure_dirs()
    await init_db()
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="CameraGestures Training Server",
    description=(
        "Receives labelled HandFilm examples from the iOS ModelTrainingApp, "
        "trains gesture recognition models, and serves the resulting .tflite files."
    ),
    version=SERVER_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(examples.router)
app.include_router(training.router)
app.include_router(model.router)
app.include_router(admin.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/version", tags=["meta"])
async def version() -> dict:
    return {"version": SERVER_VERSION}
