import json
import logging
import logging.handlers
import os
import sqlite3
import time
import uuid
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import base64
import truststore
import requests as http_requests

# Inject macOS system trust store into Python's SSL to fix certificate errors
truststore.inject_into_ssl()
http_requests = http_requests.Session()

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import face_recognition as _fr
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

def _make_handler(filename: str) -> logging.handlers.RotatingFileHandler:
    """Return a RotatingFileHandler writing to LOG_DIR/<filename>."""
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=5 * 1024 * 1024,   # rotate at 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    return handler

# --- Console handler (still prints to terminal) ---
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))

# --- app logger (overall flow — primary investigation log) ---
app_log = logging.getLogger("app")
app_log.setLevel(logging.INFO)
app_log.addHandler(_console)
app_log.addHandler(_make_handler("app.log"))
app_log.propagate = False

# --- audio_extraction logger ---
extraction_log = logging.getLogger("app.extraction")
extraction_log.setLevel(logging.INFO)
extraction_log.addHandler(_make_handler("audio_extraction.log"))
# propagates to app_log automatically (app.log also receives these)

# --- transcription logger ---
transcription_log = logging.getLogger("app.transcription")
transcription_log.setLevel(logging.INFO)
transcription_log.addHandler(_make_handler("transcription.log"))

# --- summarisation logger ---
summarisation_log = logging.getLogger("app.summarisation")
summarisation_log.setLevel(logging.INFO)
summarisation_log.addHandler(_make_handler("summarisation.log"))

# --- face / cast analysis logger ---
face_log = logging.getLogger("app.face")
face_log.setLevel(logging.INFO)
face_log.addHandler(_make_handler("face_analysis.log"))

app_log.info("=" * 60)
app_log.info("Video Summariser starting up")
app_log.info("Log directory: %s", LOG_DIR)
app_log.info("=" * 60)


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------
app = FastAPI(title="Video Summariser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every HTTP request and its response to app.log."""
    start = time.perf_counter()
    app_log.info("REQUEST  %s %s  client=%s",
                 request.method, request.url.path,
                 request.client.host if request.client else "unknown")
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    app_log.log(level, "RESPONSE %s %s  status=%d  %.0fms",
                request.method, request.url.path, response.status_code, elapsed)
    return response


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    app_log.critical("OPENAI_API_KEY is not set — all API calls will fail")
else:
    app_log.info("OpenAI client initialised (key ending ...%s)", _api_key[-4:])

client = OpenAI(api_key=_api_key)

SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
WHISPER_MAX_BYTES = 25 * 1024 * 1024  # 25 MB — Whisper API hard limit
CHUNK_SECONDS     = 600               # 10-minute chunks

# Face / cast analysis
FRAME_INTERVAL  = 10    # extract one frame every N seconds
MAX_CAST        = 20    # top N TMDB cast members to return
TMDB_API_BASE   = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

_tmdb_key = os.getenv("TMDB_API_KEY")
if not _tmdb_key:
    app_log.warning("TMDB_API_KEY not set — show cast lookup will be unavailable")
else:
    app_log.info("TMDB client ready (key ending ...%s)", _tmdb_key[-4:])


# ---------------------------------------------------------------------------
# SQLite cache helpers
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "video_cache.db"


def _db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_cache (
                filename            TEXT PRIMARY KEY,
                job_id              TEXT NOT NULL,
                transcript          TEXT,
                transcript_preview  TEXT,
                summary             TEXT,
                cast_json           TEXT,
                system_prompt       TEXT,
                model               TEXT,
                max_tokens          INTEGER,
                max_words           INTEGER,
                show_title          TEXT,
                steps_completed     TEXT DEFAULT '[]',
                created_at          TEXT,
                updated_at          TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_shows (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                title_lower TEXT NOT NULL,
                created_at  TEXT,
                updated_at  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_cast (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                show_id         INTEGER NOT NULL REFERENCES manual_shows(id) ON DELETE CASCADE,
                actor_name      TEXT NOT NULL,
                character_name  TEXT,
                photo_data      TEXT,
                created_at      TEXT,
                updated_at      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_types (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL UNIQUE,
                reverse_name TEXT NOT NULL,
                created_at   TEXT,
                updated_at   TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS character_relationships (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                show_id     INTEGER NOT NULL REFERENCES manual_shows(id) ON DELETE CASCADE,
                char_a_id   INTEGER NOT NULL REFERENCES manual_cast(id) ON DELETE CASCADE,
                char_b_id   INTEGER NOT NULL REFERENCES manual_cast(id) ON DELETE CASCADE,
                rel_type_id INTEGER NOT NULL REFERENCES relationship_types(id),
                created_at  TEXT,
                UNIQUE(show_id, char_a_id, char_b_id)
            )
        """)
        # Seed default relationship types if none exist
        count = conn.execute("SELECT COUNT(*) FROM relationship_types").fetchone()[0]
        if count == 0:
            now = datetime.now(timezone.utc).isoformat()
            defaults = [
                ("Friend",  "Friend"),
                ("Father",  "Child"),
                ("Mother",  "Child"),
                ("Brother", "Brother"),
                ("Sister",  "Sister"),
                ("Lover",   "Lover"),
                ("Spouse",  "Spouse"),
                ("Child",   "Parent"),
                ("Parent",  "Child"),
                ("Enemy",   "Enemy"),
            ]
            conn.executemany(
                "INSERT INTO relationship_types (name, reverse_name, created_at, updated_at) VALUES (?,?,?,?)",
                [(n, r, now, now) for n, r in defaults],
            )


_init_db()
app_log.info("SQLite cache initialised at %s", DB_PATH)


def _get_cache(filename: str) -> dict | None:
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM video_cache WHERE filename=?", (filename,)
        ).fetchone()
    return dict(row) if row else None


def _upsert_cache(filename: str, **fields):
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        existing = conn.execute(
            "SELECT 1 FROM video_cache WHERE filename=?", (filename,)
        ).fetchone()
        if existing:
            if fields:
                set_clause = ", ".join(f"{k}=?" for k in fields)
                conn.execute(
                    f"UPDATE video_cache SET {set_clause}, updated_at=? WHERE filename=?",
                    [*fields.values(), now, filename],
                )
        else:
            all_fields = {
                "filename": filename,
                "steps_completed": "[]",
                "created_at": now,
                "updated_at": now,
                **fields,
            }
            cols = ", ".join(all_fields.keys())
            placeholders = ", ".join("?" * len(all_fields))
            conn.execute(
                f"INSERT INTO video_cache ({cols}) VALUES ({placeholders})",
                list(all_fields.values()),
            )


# ---------------------------------------------------------------------------
# Manual cast DB helpers
# ---------------------------------------------------------------------------

def _list_manual_shows() -> list[dict]:
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM manual_shows ORDER BY title_lower"
        ).fetchall()
    return [dict(r) for r in rows]


def _search_manual_shows(query: str) -> list[dict]:
    pattern = f"%{query.lower()}%"
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM manual_shows WHERE title_lower LIKE ? ORDER BY title_lower",
            (pattern,),
        ).fetchall()
    return [dict(r) for r in rows]


def _get_manual_show_by_title(title: str) -> dict | None:
    """Exact case-insensitive title lookup."""
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id FROM manual_shows WHERE title_lower=?", (title.strip().lower(),)
        ).fetchone()
    return _get_manual_show(row["id"]) if row else None


def _sync_tmdb_to_manual_db(show_title: str, tmdb_cast: list[dict], job_id: str) -> int:
    """
    Persist TMDB cast data into the manual DB.
    TMDB photo URLs are stored as-is (no download); _actor_feature handles both
    URLs and base64 data URIs.
    Returns the new show_id.
    """
    show_id = _create_manual_show(show_title)
    for member in tmdb_cast:
        _add_manual_cast_member(
            show_id,
            actor_name=member.get("actor_name") or "",
            character_name=member.get("character_name") or "",
            photo_data=member.get("thumbnail"),   # TMDB URL stored directly
        )
    face_log.info(
        "[%s] Synced %d TMDB cast members to manual DB  show='%s' id=%d",
        job_id, len(tmdb_cast), show_title, show_id,
    )
    return show_id


def _get_manual_show(show_id: int) -> dict | None:
    with _db_conn() as conn:
        show = conn.execute(
            "SELECT id, title, created_at, updated_at FROM manual_shows WHERE id=?", (show_id,)
        ).fetchone()
        if not show:
            return None
        cast = conn.execute(
            "SELECT id, actor_name, character_name, photo_data, created_at, updated_at "
            "FROM manual_cast WHERE show_id=? ORDER BY id",
            (show_id,),
        ).fetchall()
    return {**dict(show), "cast": [dict(m) for m in cast]}


def _create_manual_show(title: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO manual_shows (title, title_lower, created_at, updated_at) VALUES (?,?,?,?)",
            (title.strip(), title.strip().lower(), now, now),
        )
        return cur.lastrowid


def _add_manual_cast_member(
    show_id: int, actor_name: str, character_name: str, photo_data: str | None
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO manual_cast (show_id, actor_name, character_name, photo_data, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (show_id, actor_name.strip(), character_name.strip() if character_name else None, photo_data, now, now),
        )
        return cur.lastrowid


def _update_manual_cast_member(cast_id: int, **fields) -> bool:
    if not fields:
        return False
    now = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    with _db_conn() as conn:
        cur = conn.execute(
            f"UPDATE manual_cast SET {set_clause}, updated_at=? WHERE id=?",
            [*fields.values(), now, cast_id],
        )
    return cur.rowcount > 0


def _delete_manual_cast_member(cast_id: int) -> bool:
    with _db_conn() as conn:
        cur = conn.execute("DELETE FROM manual_cast WHERE id=?", (cast_id,))
    return cur.rowcount > 0


def _delete_manual_show(show_id: int) -> bool:
    with _db_conn() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.execute("DELETE FROM manual_shows WHERE id=?", (show_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Relationship DB helpers
# ---------------------------------------------------------------------------

def _list_relationship_types() -> list[dict]:
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, reverse_name FROM relationship_types ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def _add_relationship_type(name: str, reverse_name: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO relationship_types (name, reverse_name, created_at, updated_at) VALUES (?,?,?,?)",
            (name.strip(), reverse_name.strip(), now, now),
        )
        return cur.lastrowid


def _get_relationships_for_show(show_id: int) -> list[dict]:
    with _db_conn() as conn:
        rows = conn.execute("""
            SELECT
                cr.id,
                cr.char_a_id,
                cr.char_b_id,
                ca.character_name AS char_a_name,
                ca.actor_name     AS char_a_actor,
                ca.photo_data     AS char_a_photo,
                cb.character_name AS char_b_name,
                cb.actor_name     AS char_b_actor,
                cb.photo_data     AS char_b_photo,
                rt.name           AS rel_type
            FROM character_relationships cr
            JOIN manual_cast ca ON ca.id = cr.char_a_id
            JOIN manual_cast cb ON cb.id = cr.char_b_id
            JOIN relationship_types rt ON rt.id = cr.rel_type_id
            WHERE cr.show_id = ?
            ORDER BY ca.character_name, cb.character_name
        """, (show_id,)).fetchall()
    return [dict(r) for r in rows]


def _set_relationship(show_id: int, char_a_id: int, char_b_id: int, rel_type_name: str):
    with _db_conn() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        fwd = conn.execute(
            "SELECT id, reverse_name FROM relationship_types WHERE name=?", (rel_type_name,)
        ).fetchone()
        if not fwd:
            raise ValueError(f"Unknown relationship type: {rel_type_name!r}")
        fwd_id = fwd["id"]
        rev_name = fwd["reverse_name"]
        rev = conn.execute(
            "SELECT id FROM relationship_types WHERE name=?", (rev_name,)
        ).fetchone()
        rev_id = rev["id"] if rev else fwd_id

        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO character_relationships (show_id, char_a_id, char_b_id, rel_type_id, created_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(show_id, char_a_id, char_b_id) DO UPDATE SET rel_type_id=excluded.rel_type_id
        """, (show_id, char_a_id, char_b_id, fwd_id, now))
        conn.execute("""
            INSERT INTO character_relationships (show_id, char_a_id, char_b_id, rel_type_id, created_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(show_id, char_a_id, char_b_id) DO UPDATE SET rel_type_id=excluded.rel_type_id
        """, (show_id, char_b_id, char_a_id, rev_id, now))


def _delete_relationship(show_id: int, char_a_id: int, char_b_id: int) -> bool:
    with _db_conn() as conn:
        cur = conn.execute(
            """DELETE FROM character_relationships
               WHERE show_id=?
               AND ((char_a_id=? AND char_b_id=?) OR (char_a_id=? AND char_b_id=?))""",
            (show_id, char_a_id, char_b_id, char_b_id, char_a_id),
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Stage 1 — Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, audio_path: str, job_id: str) -> None:
    """Extract audio track from video using FFmpeg."""
    extraction_log.info("[%s] Starting audio extraction", job_id)
    extraction_log.info("[%s] Source video : %s", job_id, Path(video_path).name)
    extraction_log.info("[%s] Target audio : %s", job_id, Path(audio_path).name)
    extraction_log.info("[%s] FFmpeg params : 16 kHz, mono, pcm_s16le", job_id)

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        extraction_log.critical("[%s] FFmpeg FAILED (exit code %d) after %.1fs",
                                job_id, result.returncode, elapsed)
        extraction_log.critical("[%s] FFmpeg stderr: %s", job_id, result.stderr[-500:])
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

    size_mb = os.path.getsize(audio_path) / 1024 / 1024
    extraction_log.info("[%s] Extraction complete in %.1fs", job_id, elapsed)
    extraction_log.info("[%s] Output audio size: %.2f MB", job_id, size_mb)


def _split_audio(audio_path: str, tmp_dir: str, job_id: str) -> list[str]:
    """Split a WAV file into CHUNK_SECONDS-long pieces."""
    chunk_pattern = os.path.join(tmp_dir, "chunk_%03d.wav")
    extraction_log.info("[%s] Audio exceeds 25 MB — splitting into %ds chunks",
                        job_id, CHUNK_SECONDS)

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-f", "segment",
            "-segment_time", str(CHUNK_SECONDS),
            "-c", "copy",
            chunk_pattern,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        extraction_log.critical("[%s] FFmpeg split FAILED (exit %d): %s",
                                job_id, result.returncode, result.stderr[-300:])
        raise RuntimeError(f"FFmpeg split error: {result.stderr}")

    chunks = sorted(str(p) for p in Path(tmp_dir).glob("chunk_*.wav"))
    for i, c in enumerate(chunks, 1):
        extraction_log.info("[%s] Chunk %d: %s (%.2f MB)",
                            job_id, i, Path(c).name,
                            os.path.getsize(c) / 1024 / 1024)
    extraction_log.info("[%s] Split produced %d chunk(s)", job_id, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Stage 2 — Face / Cast Analysis
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, frames_dir: str, job_id: str) -> list[str]:
    """Extract one JPEG frame every FRAME_INTERVAL seconds using FFmpeg."""
    face_log.info("[%s] Extracting frames (1 per %ds)", job_id, FRAME_INTERVAL)
    output_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps=1/{FRAME_INTERVAL}",
            "-q:v", "2",
            output_pattern,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        face_log.error("[%s] FFmpeg frame extraction failed: %s",
                       job_id, result.stderr[-300:])
        raise RuntimeError(f"Frame extraction failed: {result.stderr}")
    frames = sorted(str(p) for p in Path(frames_dir).glob("frame_*.jpg"))
    face_log.info("[%s] Extracted %d frames", job_id, len(frames))
    return frames


def _face_histogram(crop) -> object:
    """Normalised 3-channel colour histogram for a BGR face crop."""
    small = cv2.resize(crop, (32, 32))
    hist  = cv2.calcHist([small], [0, 1, 2], None,
                         [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def detect_faces_in_frame(img_path: str) -> list:
    """
    Return a list of (bgr_crop, encoding) tuples for each face detected.
    Uses face_recognition (HOG model) when available, falls back to Haar cascade.
    encoding is a 128-d numpy array or None in the fallback path.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return []

    if _FR_AVAILABLE:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locations = _fr.face_locations(img_rgb, model="hog")
        if not locations:
            return []
        encodings = _fr.face_encodings(img_rgb, locations)
        return [
            (img_bgr[top:bottom, left:right], enc)
            for (top, right, bottom, left), enc in zip(locations, encodings)
        ]
    else:
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        detected = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if not len(detected):
            return []
        return [(img_bgr[y:y+h, x:x+w], None) for (x, y, w, h) in detected]


def cluster_face_crops(faces: list, tolerance: float = 0.55) -> list[list[int]]:
    """
    Cluster faces by identity.
    faces: list of (bgr_crop, encoding|None) tuples from detect_faces_in_frame.
    Uses face embedding distance (average-link) when face_recognition is available,
    falls back to colour histogram correlation otherwise.
    Returns a list of clusters; each cluster is a list of indices into faces.
    """
    if not faces:
        return []

    if _FR_AVAILABLE and faces[0][1] is not None:
        encodings = [enc for _, enc in faces]
        assigned  = [False] * len(encodings)
        clusters: list[list[int]] = []
        for i in range(len(encodings)):
            if assigned[i]:
                continue
            cluster     = [i]
            assigned[i] = True
            for j in range(i + 1, len(encodings)):
                if assigned[j]:
                    continue
                cluster_encs = [encodings[k] for k in cluster]
                avg_dist = float(_fr.face_distance(cluster_encs, encodings[j]).mean())
                if avg_dist <= tolerance:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters
    else:
        crops    = [c for c, _ in faces]
        hists    = [_face_histogram(c) for c in crops]
        assigned = [False] * len(crops)
        clusters = []
        for i in range(len(crops)):
            if assigned[i]:
                continue
            cluster     = [i]
            assigned[i] = True
            for j in range(i + 1, len(crops)):
                if assigned[j]:
                    continue
                score = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL)
                if score >= 0.80:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters


def _best_rep_index(cluster: list[int], all_encodings: list) -> int:
    """
    Return the index (into all_encodings) of the most central face in a cluster.
    Picks the face with the smallest average distance to all other cluster members.
    Falls back to the middle element when encodings are unavailable.
    """
    if len(cluster) == 1 or all_encodings[cluster[0]] is None:
        return cluster[len(cluster) // 2]
    cluster_encs = [all_encodings[i] for i in cluster]
    best_pos, best_avg = 0, float("inf")
    for pos, enc in enumerate(cluster_encs):
        others = cluster_encs[:pos] + cluster_encs[pos + 1:]
        if others:
            avg_dist = float(_fr.face_distance(others, enc).mean())
            if avg_dist < best_avg:
                best_avg, best_pos = avg_dist, pos
    return cluster[best_pos]


def _crop_to_thumbnail(crop) -> str:
    """Encode a BGR face crop as a base64 data-URI JPEG (100 × 100 px)."""
    small = cv2.resize(crop, (100, 100))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def fetch_tmdb_cast(show_title: str, job_id: str) -> list[dict]:
    """Search TMDB for *show_title* and return up to MAX_CAST cast members."""
    if not _tmdb_key:
        face_log.warning("[%s] TMDB_API_KEY missing — skipping cast lookup", job_id)
        return []

    face_log.info("[%s] Searching TMDB for '%s'", job_id, show_title)
    try:
        search = http_requests.get(
            f"{TMDB_API_BASE}/search/multi",
            params={"api_key": _tmdb_key, "query": show_title},
            timeout=10,
        )
        search.raise_for_status()
    except Exception as exc:
        face_log.error("[%s] TMDB search error: %s", job_id, exc)
        return []

    results = search.json().get("results", [])
    if not results:
        face_log.warning("[%s] No TMDB results for '%s'", job_id, show_title)
        return []

    top        = results[0]
    media_type = top.get("media_type", "movie")
    result_id  = top.get("id")
    title      = top.get("title") or top.get("name", "Unknown")
    face_log.info("[%s] TMDB top hit: %s (id=%s, type=%s)",
                  job_id, title, result_id, media_type)

    credits_url = (
        f"{TMDB_API_BASE}/tv/{result_id}/aggregate_credits"
        if media_type == "tv"
        else f"{TMDB_API_BASE}/movie/{result_id}/aggregate_credits"
    )
    try:
        credits_resp = http_requests.get(
            credits_url, params={"api_key": _tmdb_key}, timeout=10
        )
        credits_resp.raise_for_status()
    except Exception as exc:
        face_log.error("[%s] TMDB credits error: %s", job_id, exc)
        return []

    cast = credits_resp.json().get("cast", [])[:MAX_CAST]
    face_log.info("[%s] TMDB returned %d cast members", job_id, len(cast))
    return [
        {
            "actor_name":     m.get("name"),
            "character_name": m.get("character"),
            "tmdb_id":        m.get("id"),
            "thumbnail":      f"{TMDB_IMAGE_BASE}{m['profile_path']}"
                              if m.get("profile_path") else None,
            "order":          m.get("order", 999),
        }
        for m in cast
    ]


def _actor_feature(url_or_data: str, job_id: str):
    """
    Load an actor photo (URL or base64 data URI) and return a face feature vector.
    Returns a 128-d face encoding when face_recognition is available,
    or a colour histogram as a fallback.  Returns None on any error.
    """
    try:
        if url_or_data.startswith("data:image"):
            _, b64 = url_or_data.split(",", 1)
            img_bytes = base64.b64decode(b64)
        else:
            resp = http_requests.get(url_or_data, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        if _FR_AVAILABLE:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            encs    = _fr.face_encodings(img_rgb)
            return encs[0] if encs else None
        else:
            gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            detected = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(detected):
                x, y, w, h = detected[0]
                crop = img_bgr[y:y+h, x:x+w]
            else:
                h, w = img_bgr.shape[:2]
                side = min(h, w)
                y0   = (h - side) // 2
                x0   = (w - side) // 2
                crop = img_bgr[y0:y0+side, x0:x0+side]
            return _face_histogram(crop)
    except Exception as exc:
        face_log.warning("[%s] Could not build feature for actor photo: %s",
                         job_id, exc)
        return None


def match_faces_to_cast(
    rep_features: list,
    tmdb_cast: list[dict],
    job_id: str,
    tolerance: float = 0.55,
) -> list:
    """
    Match each detected face to the closest TMDB cast member.
    Uses face distance (face_recognition) when available; falls back to histogram
    correlation otherwise.  Enforces one-to-one assignment so each actor is
    claimed by at most one detected face.
    Returns a parallel list of match dicts or None for unmatched faces.
    """
    if not tmdb_cast or not rep_features:
        return [None] * len(rep_features)

    face_log.info("[%s] Building actor features for %d TMDB cast members",
                  job_id, len(tmdb_cast))
    actor_features = [
        _actor_feature(m["thumbnail"], job_id) if m.get("thumbnail") else None
        for m in tmdb_cast
    ]

    use_encodings = _FR_AVAILABLE and any(f is not None for f in rep_features)
    face_log.info("[%s] Matching %d detected faces (mode=%s)",
                  job_id, len(rep_features),
                  "face-encoding" if use_encodings else "histogram")

    matched_actors: set[int] = set()
    matches = []
    for face_feat in rep_features:
        best_val   = float("inf") if use_encodings else -1.0
        best_actor = None
        best_idx   = None
        for idx, (member, actor_feat) in enumerate(zip(tmdb_cast, actor_features)):
            if actor_feat is None or idx in matched_actors:
                continue
            if use_encodings:
                dist = float(_fr.face_distance([actor_feat], face_feat)[0])
                if dist < best_val:
                    best_val, best_actor, best_idx = dist, member, idx
            else:
                score = float(cv2.compareHist(face_feat, actor_feat, cv2.HISTCMP_CORREL))
                if score > best_val:
                    best_val, best_actor, best_idx = score, member, idx

        if use_encodings:
            matched = best_actor is not None and best_val <= tolerance
            display = round(1.0 - best_val, 2) if matched else None
        else:
            matched = best_actor is not None and best_val >= tolerance
            display = round(best_val, 2) if matched else None

        if matched:
            matched_actors.add(best_idx)
            face_log.info("[%s] Match: face → %s (score=%.2f)",
                          job_id, best_actor.get("actor_name"), display)
            matches.append({
                "actor_name":     best_actor.get("actor_name"),
                "character_name": best_actor.get("character_name"),
                "match_score":    display,
            })
        else:
            face_log.info("[%s] No match for face (best=%.2f)",
                          job_id, round(1.0 - best_val if use_encodings else best_val, 2))
            matches.append(None)
    return matches


def analyse_cast(
    video_path: str,
    show_title: str,
    job_id: str,
    cast_source: str = "tmdb",
    manual_show_id: int | None = None,
) -> dict:
    """
    Extract frames, detect faces, cluster by identity, calculate screen time,
    and optionally fetch the cast list from TMDB or the manual cast database.
    """
    face_log.info("[%s] Starting cast analysis (source=%s)", job_id, cast_source)
    output: dict = {
        "frames_analysed": 0,
        "detected_faces":  [],
        "tmdb_cast":       [],
        "cast_source":     cast_source,
    }

    # _rep_features is kept at function scope so we can match against TMDB after the fetch
    _rep_features: list = []

    if not _CV2_AVAILABLE:
        face_log.warning("[%s] OpenCV not installed — face detection skipped", job_id)
    else:
        try:
            with tempfile.TemporaryDirectory() as frames_dir:
                frames = extract_frames(video_path, frames_dir, job_id)
                output["frames_analysed"] = len(frames)

                # all_faces: list of (bgr_crop, encoding|None)
                all_faces: list = []
                for fp in frames:
                    all_faces.extend(detect_faces_in_frame(fp))
                face_log.info("[%s] Total face instances detected: %d",
                              job_id, len(all_faces))

                if all_faces:
                    all_encs  = [enc for _, enc in all_faces]
                    all_crops = [c   for c, _   in all_faces]

                    clusters = cluster_face_crops(all_faces)
                    face_log.info("[%s] Unique faces after clustering: %d",
                                  job_id, len(clusters))

                    rep_indices = [_best_rep_index(c, all_encs) for c in clusters]
                    rep_crops   = [all_crops[i] for i in rep_indices]
                    _rep_features = [all_encs[i] if all_encs[i] is not None
                                     else _face_histogram(all_crops[i])
                                     for i in rep_indices]

                    for idx, cluster in enumerate(clusters):
                        rep = rep_crops[idx]
                        pct = round(len(cluster) / max(len(frames), 1) * 100, 1)
                        output["detected_faces"].append({
                            "face_id":         idx + 1,
                            "appearances":     len(cluster),
                            "screen_time_pct": pct,
                            "thumbnail":       _crop_to_thumbnail(rep),
                            "actor_name":      None,
                            "character_name":  None,
                            "match_score":     None,
                        })

                    output["detected_faces"].sort(
                        key=lambda x: x["screen_time_pct"], reverse=True
                    )
                    face_log.info("[%s] Top face: %d appearances (%.1f%%)",
                                  job_id,
                                  output["detected_faces"][0]["appearances"],
                                  output["detected_faces"][0]["screen_time_pct"])
        except Exception as exc:
            face_log.error("[%s] Face detection error (non-fatal): %s", job_id, exc)

    if cast_source == "manual" and manual_show_id:
        show_data = _get_manual_show(manual_show_id)
        if show_data:
            output["tmdb_cast"] = [
                {
                    "actor_name":     m["actor_name"],
                    "character_name": m["character_name"],
                    "tmdb_id":        m["id"],
                    "thumbnail":      m["photo_data"],
                    "order":          i,
                }
                for i, m in enumerate(show_data["cast"])
            ]
            face_log.info("[%s] Using manual cast: %d members for show '%s'",
                          job_id, len(output["tmdb_cast"]), show_data["title"])
        else:
            face_log.warning("[%s] Manual show_id=%s not found", job_id, manual_show_id)
    elif show_title:
        # Check manual DB first — TMDB data is synced here automatically on first fetch
        cached_show = _get_manual_show_by_title(show_title)
        if cached_show:
            output["tmdb_cast"] = [
                {
                    "actor_name":     m["actor_name"],
                    "character_name": m["character_name"],
                    "tmdb_id":        m["id"],
                    "thumbnail":      m["photo_data"],
                    "order":          i,
                }
                for i, m in enumerate(cached_show["cast"])
            ]
            output["cast_source"] = "tmdb_local"
            face_log.info(
                "[%s] TMDB data served from manual DB (show='%s', %d members) — skipped API call",
                job_id, cached_show["title"], len(output["tmdb_cast"]),
            )
        else:
            tmdb_cast = fetch_tmdb_cast(show_title, job_id)
            output["tmdb_cast"] = tmdb_cast
            if tmdb_cast:
                try:
                    _sync_tmdb_to_manual_db(show_title, tmdb_cast, job_id)
                except Exception as exc:
                    face_log.error("[%s] TMDB sync to manual DB failed (non-fatal): %s", job_id, exc)

    # Match detected faces against cast actor photos (if we have both)
    if _rep_features and output["tmdb_cast"]:
        # detected_faces is sorted by screen_time_pct; rebuild matching order
        # by face_id so indices align with _rep_features (which was built pre-sort)
        id_to_entry = {e["face_id"]: e for e in output["detected_faces"]}
        sorted_features = [_rep_features[fid - 1] for fid in sorted(id_to_entry)]
        matches = match_faces_to_cast(sorted_features, output["tmdb_cast"], job_id)
        for face_id, match in zip(sorted(id_to_entry), matches):
            if match:
                id_to_entry[face_id]["actor_name"]     = match["actor_name"]
                id_to_entry[face_id]["character_name"] = match["character_name"]
                id_to_entry[face_id]["match_score"]    = match["match_score"]

    face_log.info(
        "[%s] Cast analysis complete — %d unique faces, %d TMDB cast members",
        job_id, len(output["detected_faces"]), len(output["tmdb_cast"]),
    )
    return output


# ---------------------------------------------------------------------------
# Stage 3 — Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, job_id: str) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    file_size = os.path.getsize(audio_path)
    file_size_mb = file_size / 1024 / 1024

    transcription_log.info("[%s] Starting transcription", job_id)
    transcription_log.info("[%s] Audio file  : %s", job_id, Path(audio_path).name)
    transcription_log.info("[%s] Audio size  : %.2f MB", job_id, file_size_mb)
    transcription_log.info("[%s] Whisper model: whisper-1", job_id)

    if file_size <= WHISPER_MAX_BYTES:
        transcription_log.info("[%s] File within 25 MB limit — single API call", job_id)
        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            text = client.audio.transcriptions.create(model="whisper-1", file=f).text
        elapsed = time.perf_counter() - t0
        transcription_log.info("[%s] Transcription complete in %.1fs", job_id, elapsed)
        transcription_log.info("[%s] Transcript length: %d chars", job_id, len(text))
        transcription_log.info("[%s] Transcript preview: %.200s", job_id, text)
        return text

    # File too large — chunk it
    transcription_log.warning("[%s] File exceeds 25 MB (%.2f MB) — chunked transcription",
                               job_id, file_size_mb)
    with tempfile.TemporaryDirectory() as chunk_dir:
        chunks = _split_audio(audio_path, chunk_dir, job_id)
        parts: list[str] = []
        for i, chunk_path in enumerate(chunks, 1):
            chunk_mb = os.path.getsize(chunk_path) / 1024 / 1024
            transcription_log.info("[%s] Transcribing chunk %d/%d (%.2f MB)",
                                   job_id, i, len(chunks), chunk_mb)
            t0 = time.perf_counter()
            with open(chunk_path, "rb") as f:
                part = client.audio.transcriptions.create(model="whisper-1", file=f).text
            elapsed = time.perf_counter() - t0
            transcription_log.info("[%s] Chunk %d done in %.1fs — %d chars",
                                   job_id, i, elapsed, len(part))
            transcription_log.info("[%s] Chunk %d preview: %.150s", job_id, i, part)
            parts.append(part)

    full_text = " ".join(parts)
    transcription_log.info("[%s] All chunks transcribed — total %d chars", job_id, len(full_text))
    return full_text


# ---------------------------------------------------------------------------
# Stage 4 — Summarisation
# ---------------------------------------------------------------------------

def summarise_transcript(
    transcript: str,
    job_id: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    max_words: int,
) -> str:
    """Generate a summary using the provided model and parameters."""
    full_system_prompt = f"{system_prompt.strip()} Keep the summary within {max_words} words."

    summarisation_log.info("[%s] Starting summarisation", job_id)
    summarisation_log.info("[%s] Transcript length : %d chars", job_id, len(transcript))
    summarisation_log.info("[%s] Model             : %s", job_id, model)
    summarisation_log.info("[%s] Max tokens        : %d", job_id, max_tokens)
    summarisation_log.info("[%s] Max words         : %d", job_id, max_words)
    summarisation_log.info("[%s] Temperature       : 0.3", job_id)
    summarisation_log.info("[%s] System prompt     : %s", job_id, full_system_prompt)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": full_system_prompt,
            },
            {
                "role": "user",
                "content": f"Please summarise the following video transcript:\n\n{transcript}",
            },
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    elapsed = time.perf_counter() - t0
    summary = response.choices[0].message.content.strip()

    summarisation_log.info("[%s] Summary generated in %.1fs", job_id, elapsed)
    summarisation_log.info("[%s] Summary length: %d chars", job_id, len(summary))
    summarisation_log.info("[%s] Token usage — prompt: %d, completion: %d, total: %d",
                           job_id,
                           response.usage.prompt_tokens,
                           response.usage.completion_tokens,
                           response.usage.total_tokens)
    summarisation_log.info("[%s] Summary:\n%s", job_id, summary)
    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    app_log.info("Health check OK")
    return {"status": "ok"}


@app.get("/check-cache")
def check_cache(filename: str):
    norm = filename.lower()
    cached = _get_cache(norm)
    if not cached:
        return {"cached": False, "filename": filename}

    steps = json.loads(cached["steps_completed"] or "[]")
    all_steps = {"audio_extraction", "cast_analysis", "transcription", "summarisation"}
    fully_cached = all_steps.issubset(set(steps))

    result = None
    if fully_cached:
        cast_data = json.loads(cached["cast_json"]) if cached.get("cast_json") else {}
        result = {
            "job_id":             cached["job_id"],
            "filename":           filename,
            "transcript_preview": cached.get("transcript_preview"),
            "summary":            cached.get("summary"),
            "cast":               cast_data,
            "system_prompt":      cached.get("system_prompt"),
            "model":              cached.get("model"),
            "max_words":          cached.get("max_words"),
        }

    app_log.info("CACHE CHECK  filename=%s  steps=%s  fully_cached=%s",
                 norm, steps, fully_cached)
    return {
        "cached":          fully_cached,
        "filename":        filename,
        "steps_completed": steps,
        "result":          result,
    }


DEFAULT_SYSTEM_PROMPT = """\
# Role and Objective
You are a content summarization assistant. Given a video transcript, write a concise summary that captures the main topics, key points, and overall narrative. Reason internally as needed, but do not reveal internal reasoning.

# Instructions
- Use simple English to convey the story plot.
- Convey the story plot using the chararacter names. But, do not reveal the final climax of the show.
- Make the summary interesting and catchy.
- Write the summary so it makes readers curious and interested to watch the TV show.
- Before finalizing, do a brief check that the summary is spoiler-safe, uses very simple English, and follows the required output format exactly.
- Output plain text only.

# Output Format
Return exactly two sections in this order:
Summary:
<1 concise paragraph in very simple English, written as a spoiler-safe preview>
Keywords:
- <keyword or short phrase 1>
- <keyword or short phrase 2>
- <keyword or short phrase 3>
- <keyword or short phrase 4>
- <keyword or short phrase 5>
Do not add any other sections, labels, commentary, or formatting.

# Fallback Condition
If the transcript is missing, incomplete, or too short to summarize safely without guessing or revealing spoilers, return:
Summary:
Not enough transcript content to create a safe summary.
Keywords:
- insufficient transcript
- video unavailable
- missing context
- incomplete transcript
- spoiler-safe summary\
"""

@app.post("/summarise")
async def summarise_video(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o-mini"),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    max_tokens: int = Form(500),
    max_words: int = Form(200),
    show_title: str = Form(""),
    features: str = Form("{}"),
    cast_source: str = Form("tmdb"),
    manual_show_id: int = Form(0),
):
    suffix = Path(file.filename).suffix.lower()
    job_id = str(uuid.uuid4())

    try:
        flags: dict = json.loads(features)
    except json.JSONDecodeError:
        flags = {}

    app_log.info("=" * 60)
    app_log.info("JOB %s  START", job_id)
    app_log.info("JOB %s  file=%s  format=%s", job_id, file.filename, suffix)
    app_log.info("JOB %s  feature flags: %s", job_id, flags)
    if show_title:
        app_log.info("JOB %s  show_title=%s", job_id, show_title)

    if suffix not in SUPPORTED_FORMATS:
        app_log.warning("JOB %s  REJECTED — unsupported format '%s'", job_id, suffix)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    # Load cache
    norm_filename = file.filename.lower()
    cached = _get_cache(norm_filename)
    steps_done: set[str] = set(json.loads(cached["steps_completed"] or "[]")) if cached else set()
    if steps_done:
        app_log.info("JOB %s  CACHE HIT  steps_done=%s", job_id, sorted(steps_done))

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, f"{job_id}{suffix}")
        audio_path = os.path.join(tmp_dir, f"{job_id}.wav")

        contents = await file.read()
        video_size_mb = len(contents) / 1024 / 1024
        with open(video_path, "wb") as f:
            f.write(contents)
        app_log.info("JOB %s  video saved — %.2f MB", job_id, video_size_mb)

        run_audio = flags.get("summarisation", True)

        transcript = None
        summary    = None
        cast_data  = None

        # Stage 1/4 — Audio extraction
        if "transcription" in steps_done:
            transcript = cached["transcript"]
            app_log.info("JOB %s  STAGE 1/4: audio extraction  SKIPPED (cached)", job_id)
        elif run_audio:
            app_log.info("JOB %s  STAGE 1/4: audio extraction  →  see audio_extraction.log", job_id)
            try:
                extract_audio(video_path, audio_path, job_id)
                steps_done.add("audio_extraction")
                _upsert_cache(norm_filename, job_id=job_id,
                              steps_completed=json.dumps(sorted(steps_done)))
                app_log.info("JOB %s  STAGE 1/4: audio extraction  DONE", job_id)
            except RuntimeError as e:
                app_log.critical("JOB %s  STAGE 1/4 FAILED: %s", job_id, e)
                raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        else:
            app_log.info("JOB %s  STAGE 1/4: audio extraction  SKIPPED (summarisation flag off)", job_id)

        # Stage 2/4 — Face / cast analysis
        if "cast_analysis" in steps_done:
            cast_data = json.loads(cached["cast_json"]) if cached.get("cast_json") else {}
            app_log.info("JOB %s  STAGE 2/4: cast analysis  SKIPPED (cached)", job_id)

            # Opportunistic TMDB sync: the previous run may have cached an empty
            # tmdb_cast (e.g. due to SSL errors).  If the show isn't in the manual DB
            # yet, try fetching TMDB now and sync — then update the video cache too.
            if cast_source == "tmdb" and show_title and not _get_manual_show_by_title(show_title):
                cached_tmdb = cast_data.get("tmdb_cast") or []
                if cached_tmdb:
                    # Cache already has TMDB data — just backfill the manual DB
                    try:
                        _sync_tmdb_to_manual_db(show_title, cached_tmdb, job_id)
                        app_log.info("JOB %s  Backfilled manual DB from cache for '%s'",
                                     job_id, show_title)
                    except Exception as exc:
                        app_log.warning("JOB %s  Backfill failed (non-fatal): %s", job_id, exc)
                else:
                    # Cache has no TMDB data — fetch fresh and sync
                    try:
                        fresh_tmdb = fetch_tmdb_cast(show_title, job_id)
                        if fresh_tmdb:
                            _sync_tmdb_to_manual_db(show_title, fresh_tmdb, job_id)
                            cast_data["tmdb_cast"] = fresh_tmdb
                            _upsert_cache(norm_filename, cast_json=json.dumps(cast_data))
                            app_log.info(
                                "JOB %s  Re-fetched TMDB and synced %d members for '%s'",
                                job_id, len(fresh_tmdb), show_title,
                            )
                    except Exception as exc:
                        app_log.warning("JOB %s  TMDB re-fetch failed (non-fatal): %s", job_id, exc)
        else:
            app_log.info("JOB %s  STAGE 2/4: cast analysis  →  see face_analysis.log", job_id)
            _mid = manual_show_id if manual_show_id else None
            cast_data = analyse_cast(video_path, show_title, job_id,
                                     cast_source=cast_source, manual_show_id=_mid)
            steps_done.add("cast_analysis")
            _upsert_cache(norm_filename, job_id=job_id,
                          cast_json=json.dumps(cast_data),
                          show_title=show_title,
                          steps_completed=json.dumps(sorted(steps_done)))
            app_log.info(
                "JOB %s  STAGE 2/4: cast analysis  DONE "
                "(%d faces, %d TMDB cast)",
                job_id,
                len(cast_data["detected_faces"]),
                len(cast_data["tmdb_cast"]),
            )

        if "transcription" in steps_done:
            # transcript already set above; skip Stage 3
            app_log.info("JOB %s  STAGE 3/4: transcription  SKIPPED (cached)", job_id)
        elif run_audio:
            # Stage 3/4 — Transcription
            app_log.info("JOB %s  STAGE 3/4: transcription  →  see transcription.log", job_id)
            try:
                transcript = transcribe_audio(audio_path, job_id)
                steps_done.add("transcription")
                preview = transcript[:500] + ("..." if len(transcript) > 500 else "")
                _upsert_cache(norm_filename, job_id=job_id,
                              transcript=transcript,
                              transcript_preview=preview,
                              steps_completed=json.dumps(sorted(steps_done)))
                app_log.info("JOB %s  STAGE 3/4: transcription  DONE (%d chars)", job_id, len(transcript))
            except Exception as e:
                app_log.critical("JOB %s  STAGE 3/4 FAILED: %s", job_id, e)
                raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
        else:
            app_log.info("JOB %s  STAGES 3+4/4: transcription + summarisation  SKIPPED (flag off)", job_id)

        if "summarisation" in steps_done:
            summary = cached["summary"]
            app_log.info("JOB %s  STAGE 4/4: summarisation  SKIPPED (cached)", job_id)
        elif run_audio and transcript:
            # Stage 4/4 — Summarisation
            app_log.info("JOB %s  STAGE 4/4: summarisation  →  see summarisation.log", job_id)
            try:
                summary = summarise_transcript(
                    transcript, job_id, model, system_prompt, max_tokens, max_words
                )
                steps_done.add("summarisation")
                _upsert_cache(norm_filename, job_id=job_id,
                              summary=summary,
                              system_prompt=system_prompt,
                              model=model,
                              max_tokens=max_tokens,
                              max_words=max_words,
                              steps_completed=json.dumps(sorted(steps_done)))
                app_log.info("JOB %s  STAGE 4/4: summarisation  DONE (%d chars)", job_id, len(summary))
            except Exception as e:
                app_log.critical("JOB %s  STAGE 4/4 FAILED: %s", job_id, e)
                raise HTTPException(status_code=500, detail=f"Summarisation failed: {e}")

    app_log.info("JOB %s  COMPLETE", job_id)
    app_log.info("=" * 60)

    transcript_preview = (
        transcript[:500] + ("..." if len(transcript) > 500 else "")
        if transcript else None
    )

    return JSONResponse(
        content={
            "job_id":             job_id,
            "filename":           file.filename,
            "transcript_preview": transcript_preview,
            "summary":            summary,
            "cast":               cast_data,
            "system_prompt":      system_prompt,
            "model":              model,
            "max_words":          max_words,
        }
    )


@app.post("/regenerate-summary")
async def regenerate_summary(
    filename:      str = Form(...),
    model:         str = Form("gpt-4o-mini"),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    max_tokens:    int = Form(500),
    max_words:     int = Form(200),
):
    norm_filename = filename.lower()
    cached = _get_cache(norm_filename)
    if not cached or not cached.get("transcript"):
        raise HTTPException(
            status_code=404,
            detail="No cached transcript found for this filename. Process the video first.",
        )

    job_id = str(uuid.uuid4())
    app_log.info("REGEN %s  file=%s  model=%s", job_id, norm_filename, model)

    try:
        summary = summarise_transcript(
            cached["transcript"], job_id, model, system_prompt, max_tokens, max_words
        )
    except Exception as e:
        app_log.critical("REGEN %s  FAILED: %s", job_id, e)
        raise HTTPException(status_code=500, detail=f"Summarisation failed: {e}")

    _upsert_cache(norm_filename,
                  summary=summary,
                  system_prompt=system_prompt,
                  model=model,
                  max_tokens=max_tokens,
                  max_words=max_words)

    app_log.info("REGEN %s  DONE (%d chars)", job_id, len(summary))
    return JSONResponse(content={"summary": summary, "system_prompt": system_prompt})


# ---------------------------------------------------------------------------
# Manual cast endpoints
# ---------------------------------------------------------------------------

@app.get("/manual-shows")
def list_manual_shows():
    return _list_manual_shows()


@app.get("/manual-shows/search")
def search_manual_shows(q: str = ""):
    if not q.strip():
        return []
    return _search_manual_shows(q)


@app.get("/manual-shows/{show_id}")
def get_manual_show(show_id: int):
    show = _get_manual_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="Show not found")
    return show


@app.post("/manual-shows")
async def create_manual_show(title: str = Form(...)):
    if not title.strip():
        raise HTTPException(status_code=400, detail="Title is required")
    show_id = _create_manual_show(title)
    app_log.info("MANUAL SHOW created: id=%d title=%s", show_id, title)
    return {"id": show_id, "title": title.strip()}


@app.delete("/manual-shows/{show_id}")
def delete_manual_show(show_id: int):
    ok = _delete_manual_show(show_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Show not found")
    app_log.info("MANUAL SHOW deleted: id=%d", show_id)
    return {"deleted": True}


@app.post("/manual-shows/{show_id}/cast")
async def add_cast_member(
    show_id: int,
    actor_name:     str        = Form(...),
    character_name: str        = Form(""),
    photo:          UploadFile = File(None),
    photo_url:      str        = Form(""),
):
    show = _get_manual_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="Show not found")

    photo_data = None
    if photo and photo.filename:
        content = await photo.read()
        ext = Path(photo.filename).suffix.lower().lstrip(".")
        mime = f"image/{ext}" if ext in ("jpg", "jpeg", "png", "webp", "gif") else "image/jpeg"
        if ext == "jpg":
            mime = "image/jpeg"
        b64 = base64.b64encode(content).decode("utf-8")
        photo_data = f"data:{mime};base64,{b64}"
    elif photo_url:
        if photo_url.startswith("data:"):
            photo_data = photo_url
        else:
            try:
                resp = http_requests.get(photo_url, timeout=5)
                resp.raise_for_status()
                mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
                b64 = base64.b64encode(resp.content).decode("utf-8")
                photo_data = f"data:{mime};base64,{b64}"
            except Exception:
                photo_data = None

    cast_id = _add_manual_cast_member(show_id, actor_name, character_name, photo_data)
    app_log.info("MANUAL CAST added: cast_id=%d show_id=%d actor=%s", cast_id, show_id, actor_name)
    return {"id": cast_id, "show_id": show_id, "actor_name": actor_name,
            "character_name": character_name, "photo_data": photo_data}


@app.put("/manual-shows/{show_id}/cast/{cast_id}")
async def update_cast_member(
    show_id: int,
    cast_id: int,
    actor_name:     str        = Form(None),
    character_name: str        = Form(None),
    photo:          UploadFile = File(None),
):
    fields: dict = {}
    if actor_name is not None:
        fields["actor_name"] = actor_name.strip()
    if character_name is not None:
        fields["character_name"] = character_name.strip()
    if photo and photo.filename:
        content = await photo.read()
        ext = Path(photo.filename).suffix.lower().lstrip(".")
        mime = f"image/{ext}" if ext in ("jpg", "jpeg", "png", "webp", "gif") else "image/jpeg"
        if ext == "jpg":
            mime = "image/jpeg"
        b64 = base64.b64encode(content).decode("utf-8")
        fields["photo_data"] = f"data:{mime};base64,{b64}"

    ok = _update_manual_cast_member(cast_id, **fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Cast member not found")
    app_log.info("MANUAL CAST updated: cast_id=%d", cast_id)
    return {"updated": True, **fields}


@app.delete("/manual-shows/{show_id}/cast/{cast_id}")
def delete_cast_member(show_id: int, cast_id: int):
    ok = _delete_manual_cast_member(cast_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Cast member not found")
    app_log.info("MANUAL CAST deleted: cast_id=%d", cast_id)
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Relationship type endpoints
# ---------------------------------------------------------------------------

@app.get("/relationship-types")
def list_relationship_types():
    return _list_relationship_types()


@app.post("/relationship-types")
async def create_relationship_type(
    name: str = Form(...),
    reverse_name: str = Form(...),
):
    if not name.strip() or not reverse_name.strip():
        raise HTTPException(status_code=400, detail="name and reverse_name are required")
    try:
        new_id = _add_relationship_type(name, reverse_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    app_log.info("RELATIONSHIP TYPE created: id=%d name=%s", new_id, name)
    return {"id": new_id, "name": name.strip(), "reverse_name": reverse_name.strip()}


# ---------------------------------------------------------------------------
# Character relationship endpoints
# ---------------------------------------------------------------------------

@app.get("/manual-shows/{show_id}/relationships")
def list_relationships(show_id: int):
    show = _get_manual_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="Show not found")
    return _get_relationships_for_show(show_id)


@app.post("/manual-shows/{show_id}/relationships")
async def set_relationship(
    show_id: int,
    char_a_id: int = Form(...),
    char_b_id: int = Form(...),
    rel_type_name: str = Form(...),
):
    show = _get_manual_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="Show not found")
    try:
        _set_relationship(show_id, char_a_id, char_b_id, rel_type_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    app_log.info(
        "RELATIONSHIP set: show_id=%d a=%d b=%d type=%s", show_id, char_a_id, char_b_id, rel_type_name
    )
    return {"set": True}


@app.delete("/manual-shows/{show_id}/relationships/{char_a_id}/{char_b_id}")
def delete_relationship(show_id: int, char_a_id: int, char_b_id: int):
    ok = _delete_relationship(show_id, char_a_id, char_b_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Relationship not found")
    app_log.info(
        "RELATIONSHIP deleted: show_id=%d a=%d b=%d", show_id, char_a_id, char_b_id
    )
    return {"deleted": True}
