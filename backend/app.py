from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DEFAULT_DB = Path(__file__).resolve().parent / "data" / "annotations.db"
DB_PATH = Path(os.environ.get("DPE_DB_PATH", DEFAULT_DB))


class AnnotationInput(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=2000)


class Annotation(AnnotationInput):
    debate_id: str
    username: str
    updated_at: str


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with closing(connect()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                debate_id TEXT NOT NULL,
                username TEXT NOT NULL,
                label TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (debate_id, username)
            )
            """
        )
        connection.commit()


app = FastAPI(
    title="Debate Persuasion Explorer API",
    description="SQLite persistence for HW1 graph-node annotations.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_database()


@app.get("/api/health")
def health() -> dict[str, str]:
    initialize_database()
    return {"status": "ok", "database": str(DB_PATH)}


@app.get("/api/annotations", response_model=list[Annotation])
def list_annotations(debate_id: str = Query(min_length=1, max_length=100)) -> list[dict]:
    with closing(connect()) as connection:
        rows = connection.execute(
            "SELECT debate_id, username, label, note, updated_at FROM annotations WHERE debate_id = ? ORDER BY username",
            (debate_id,),
        ).fetchall()
    return [dict(row) for row in rows]


@app.put("/api/annotations/{debate_id}/{username}", response_model=Annotation)
def upsert_annotation(debate_id: str, username: str, annotation: AnnotationInput) -> dict:
    if len(debate_id) > 100 or len(username) > 200:
        raise HTTPException(status_code=400, detail="Identifier is too long")
    updated_at = datetime.now(UTC).isoformat()
    with closing(connect()) as connection:
        connection.execute(
            """
            INSERT INTO annotations (debate_id, username, label, note, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(debate_id, username) DO UPDATE SET
                label = excluded.label,
                note = excluded.note,
                updated_at = excluded.updated_at
            """,
            (debate_id, username, annotation.label, annotation.note, updated_at),
        )
        connection.commit()
    return {
        "debate_id": debate_id,
        "username": username,
        "label": annotation.label,
        "note": annotation.note,
        "updated_at": updated_at,
    }


@app.delete("/api/annotations/{debate_id}/{username}")
def delete_annotation(debate_id: str, username: str) -> dict[str, bool]:
    with closing(connect()) as connection:
        cursor = connection.execute(
            "DELETE FROM annotations WHERE debate_id = ? AND username = ?",
            (debate_id, username),
        )
        connection.commit()
    return {"deleted": cursor.rowcount > 0}
