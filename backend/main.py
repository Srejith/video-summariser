import json
import logging
import logging.handlers
import os
import time
import uuid
import subprocess
import tempfile
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


def _actor_feature(url: str, job_id: str):
    """
    Download a TMDB profile photo and return a face feature vector.
    Returns a 128-d face encoding when face_recognition is available,
    or a colour histogram as a fallback.  Returns None on any error.
    """
    try:
        resp = http_requests.get(url, timeout=10)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
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
        face_log.warning("[%s] Could not build feature for actor photo %s: %s",
                         job_id, url, exc)
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


def analyse_cast(video_path: str, show_title: str, job_id: str) -> dict:
    """
    Extract frames, detect faces, cluster by identity, calculate screen time,
    and optionally fetch the TMDB cast list for *show_title*.
    """
    face_log.info("[%s] Starting cast analysis", job_id)
    output: dict = {
        "frames_analysed": 0,
        "detected_faces":  [],
        "tmdb_cast":       [],
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

    if show_title:
        output["tmdb_cast"] = fetch_tmdb_cast(show_title, job_id)

    # Match detected faces against TMDB actor photos (if we have both)
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


DEFAULT_SYSTEM_PROMPT = (
    "You are a content summarisation assistant. "
    "Given a video transcript, write a concise summary "
    "that captures the main topics, key points, and overall narrative."
)

@app.post("/summarise")
async def summarise_video(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o-mini"),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    max_tokens: int = Form(500),
    max_words: int = Form(200),
    show_title: str = Form(""),
    features: str = Form("{}"),
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, f"{job_id}{suffix}")
        audio_path = os.path.join(tmp_dir, f"{job_id}.wav")

        contents = await file.read()
        video_size_mb = len(contents) / 1024 / 1024
        with open(video_path, "wb") as f:
            f.write(contents)
        app_log.info("JOB %s  video saved — %.2f MB", job_id, video_size_mb)

        run_audio = flags.get("summarisation", True)

        # Stage 1/4 — Audio extraction (skipped when summarisation flag is off)
        if run_audio:
            app_log.info("JOB %s  STAGE 1/4: audio extraction  →  see audio_extraction.log", job_id)
            try:
                extract_audio(video_path, audio_path, job_id)
                app_log.info("JOB %s  STAGE 1/4: audio extraction  DONE", job_id)
            except RuntimeError as e:
                app_log.critical("JOB %s  STAGE 1/4 FAILED: %s", job_id, e)
                raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        else:
            app_log.info("JOB %s  STAGE 1/4: audio extraction  SKIPPED (summarisation flag off)", job_id)

        # Stage 2/4 — Face / cast analysis (always runs)
        app_log.info("JOB %s  STAGE 2/4: cast analysis  →  see face_analysis.log", job_id)
        cast_data = analyse_cast(video_path, show_title, job_id)
        app_log.info(
            "JOB %s  STAGE 2/4: cast analysis  DONE "
            "(%d faces, %d TMDB cast)",
            job_id,
            len(cast_data["detected_faces"]),
            len(cast_data["tmdb_cast"]),
        )

        transcript = None
        summary    = None

        if run_audio:
            # Stage 3/4 — Transcription
            app_log.info("JOB %s  STAGE 3/4: transcription  →  see transcription.log", job_id)
            try:
                transcript = transcribe_audio(audio_path, job_id)
                app_log.info("JOB %s  STAGE 3/4: transcription  DONE (%d chars)", job_id, len(transcript))
            except Exception as e:
                app_log.critical("JOB %s  STAGE 3/4 FAILED: %s", job_id, e)
                raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

            # Stage 4/4 — Summarisation
            app_log.info("JOB %s  STAGE 4/4: summarisation  →  see summarisation.log", job_id)
            try:
                summary = summarise_transcript(
                    transcript, job_id, model, system_prompt, max_tokens, max_words
                )
                app_log.info("JOB %s  STAGE 4/4: summarisation  DONE (%d chars)", job_id, len(summary))
            except Exception as e:
                app_log.critical("JOB %s  STAGE 4/4 FAILED: %s", job_id, e)
                raise HTTPException(status_code=500, detail=f"Summarisation failed: {e}")
        else:
            app_log.info("JOB %s  STAGES 3+4/4: transcription + summarisation  SKIPPED (flag off)", job_id)

    app_log.info("JOB %s  COMPLETE", job_id)
    app_log.info("=" * 60)

    transcript_preview = (
        transcript[:500] + ("..." if len(transcript) > 500 else "")
        if transcript else None
    )

    return JSONResponse(
        content={
            "job_id": job_id,
            "filename": file.filename,
            "transcript_preview": transcript_preview,
            "summary": summary,
            "cast": cast_data,
        }
    )
