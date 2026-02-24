**Framework Tool Choice**

Front end: React and Node JS
Backend: Python

**Instructions**

1. Create all the code required for this app inside the folder 'code-repository'
2. First create a plan, discuss it with me, seek approval and then start building

---

## Video Summariser — MVP

Upload a video file → audio is extracted → transcript generated → AI summary produced.

### Architecture

```
frontend/   React + Vite  (port 5173)
backend/    Python FastAPI (port 8000)
```

**Pipeline per video:**

1. **Audio extraction** — FFmpeg strips the audio track to a 16 kHz mono WAV
2. **Transcription** — OpenAI Whisper API converts speech to text
3. **Summarisation** — GPT-4o Mini produces a 2-3 paragraph summary

---

### Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg installed and on PATH (`brew install ffmpeg` on macOS)
- OpenAI API key

---

### Setup & Run

#### 1. Backend

```bash
cd code-repository/backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI key
cp .env.example .env
# Edit .env and add your real OPENAI_API_KEY

# Start the server
source .env && uvicorn main:app --reload --port 8000
```

#### 2. Frontend

```bash
cd code-repository/frontend

npm install
npm run dev
```

Open http://localhost:5173 in your browser.

---

### API

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Liveness check |
| `POST /summarise` | POST | Upload video, returns transcript preview + summary |

**POST /summarise** — multipart/form-data, field name `file`

Response:
```json
{
  "job_id": "uuid",
  "filename": "episode.mp4",
  "transcript_preview": "first 500 chars...",
  "summary": "2-3 paragraph summary"
}
```

---

### Supported Video Formats

MP4, AVI, MOV, MKV

---

### Cost per Episode (20 min video)

| Service | Cost |
|---|---|
| Whisper API | ~$0.12 |
| GPT-4o Mini | ~$0.001 |
| **Total** | **~$0.12** |
