import { useState, useRef } from "react";

const STAGES = {
  idle:        null,
  uploading:   "Uploading video...",
  extracting:  "Extracting audio...",
  analysing:   "Analysing cast...",
  transcribing:"Transcribing audio...",
  summarising: "Generating summary...",
  done:        null,
  error:       null,
};

const MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"];

// ─── Feature flags config ────────────────────────────────────────────────────
// To add a new flag: append one entry here. No other changes needed.
const FEATURE_FLAGS_CONFIG = [
  {
    key:         "summarisation",
    label:       "AI Summarisation",
    description: "Generate an AI summary from the transcript (uses GPT API)",
  },
];
const DEFAULT_FEATURE_FLAGS = Object.fromEntries(
  FEATURE_FLAGS_CONFIG.map((f) => [f.key, true])
);

const DEFAULT_SYSTEM_PROMPT =
  "You are a content summarisation assistant. " +
  "Given a video transcript, write a concise summary " +
  "that captures the main topics, key points, and overall narrative.";

export default function App() {
  const [stage, setStage]       = useState("idle");
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState(null);
  const fileInputRef = useRef(null);

  // Summarisation settings
  const [model, setModel]               = useState("gpt-4o-mini");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [maxTokens, setMaxTokens]       = useState(500);
  const [maxWords, setMaxWords]         = useState(200);
  const [showTitle, setShowTitle]       = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Feature flags
  const [featureFlags, setFeatureFlags] = useState(DEFAULT_FEATURE_FLAGS);
  const toggleFlag = (key) =>
    setFeatureFlags((prev) => ({ ...prev, [key]: !prev[key] }));

  const handleFile = async (file) => {
    if (!file) return;

    const ext = file.name.split(".").pop().toLowerCase();
    const supported = ["mp4", "avi", "mov", "mkv"];
    if (!supported.includes(ext)) {
      setError(`Unsupported format ".${ext}". Supported: ${supported.map((e) => `.${e}`).join(", ")}`);
      setStage("error");
      return;
    }

    setFileName(file.name);
    setResult(null);
    setError(null);
    setStage("uploading");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);
    formData.append("system_prompt", systemPrompt);
    formData.append("max_tokens", String(maxTokens));
    formData.append("max_words", String(maxWords));
    formData.append("show_title", showTitle);
    formData.append("features", JSON.stringify(featureFlags));

    try {
      const stageLabels = [
        "uploading", "extracting", "analysing", "transcribing",
        ...(featureFlags.summarisation ? ["summarising"] : []),
      ];
      let labelIndex = 0;
      const labelInterval = setInterval(() => {
        labelIndex = Math.min(labelIndex + 1, stageLabels.length - 1);
        setStage(stageLabels[labelIndex]);
      }, 8000);

      const response = await fetch("/summarise", {
        method: "POST",
        body: formData,
      });

      clearInterval(labelInterval);

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Processing failed");
      }

      const data = await response.json();
      setResult(data);
      setStage("done");
    } catch (err) {
      setError(err.message);
      setStage("error");
    }
  };

  const onInputChange = (e) => { handleFile(e.target.files[0]); e.target.value = ""; };
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); };
  const reset  = () => { setStage("idle"); setResult(null); setError(null); setFileName(null); };

  const isProcessing = Object.keys(STAGES)
    .filter((k) => STAGES[k] !== null && k !== "done" && k !== "error")
    .includes(stage);

  const hasCast = result?.cast && (
    result.cast.detected_faces?.length > 0 || result.cast.tmdb_cast?.length > 0
  );

  return (
    <div className="container">
      <header>
        <h1>Video Summariser</h1>
        <p className="subtitle">Upload a video to extract audio, analyse cast, and generate an AI summary</p>
      </header>

      <main>
        {stage === "idle" || stage === "error" ? (
          <>
            {/* Settings panel */}
            <div className="settings-panel">
              <button
                className="settings-toggle"
                onClick={() => setSettingsOpen((o) => !o)}
                aria-expanded={settingsOpen}
              >
                <span>Settings</span>
                <span className={`chevron ${settingsOpen ? "open" : ""}`}>›</span>
              </button>

              {settingsOpen && (
                <div className="settings-body">
                  <div className="setting-row setting-row--full">
                    <label htmlFor="show-title">Show / movie title <span className="optional">(for cast lookup)</span></label>
                    <input
                      id="show-title"
                      type="text"
                      placeholder="e.g. Breaking Bad, Inception…"
                      value={showTitle}
                      onChange={(e) => setShowTitle(e.target.value)}
                    />
                  </div>

                  <div className="setting-row">
                    <label htmlFor="model-select">Model</label>
                    <select
                      id="model-select"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                    >
                      {MODELS.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>

                  <div className="setting-row">
                    <label htmlFor="max-words">Maximum words</label>
                    <input
                      id="max-words"
                      type="number"
                      min={50} max={1000} step={50}
                      value={maxWords}
                      onChange={(e) => setMaxWords(Number(e.target.value))}
                    />
                  </div>

                  <div className="setting-row setting-row--full">
                    <label htmlFor="system-prompt">System prompt</label>
                    <textarea
                      id="system-prompt"
                      rows={4}
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                    />
                  </div>

                  <div className="setting-row">
                    <label htmlFor="max-tokens">Max tokens</label>
                    <input
                      id="max-tokens"
                      type="number"
                      min={100} max={4096} step={100}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                    />
                  </div>

                  {/* Feature flags */}
                  <div className="setting-row setting-row--full setting-row--flags">
                    <span className="flags-heading">Feature flags</span>
                    <p className="flags-note">Disable features to reduce API costs during testing.</p>
                    <div className="flags-list">
                      {FEATURE_FLAGS_CONFIG.map((flag) => (
                        <label key={flag.key} className="flag-row">
                          <span className="flag-text">
                            <span className="flag-label">{flag.label}</span>
                            <span className="flag-desc">{flag.description}</span>
                          </span>
                          <span className="toggle-switch">
                            <input
                              type="checkbox"
                              checked={featureFlags[flag.key]}
                              onChange={() => toggleFlag(flag.key)}
                            />
                            <span className="toggle-slider" />
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Drop zone */}
            <div
              className={`drop-zone ${dragOver ? "drag-over" : ""}`}
              onClick={() => fileInputRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <div className="drop-icon">🎬</div>
              <p>Drop a video file here, or <span className="link">click to browse</span></p>
              <p className="hint">Supported formats: MP4, AVI, MOV, MKV</p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv"
                onChange={onInputChange}
                hidden
              />
            </div>
            {error && <div className="error-box">Error: {error}</div>}
          </>
        ) : isProcessing ? (
          <div className="status-box">
            <div className="spinner" />
            <p className="status-label">{STAGES[stage]}</p>
            <p className="hint">{fileName}</p>
          </div>
        ) : (
          <div className="result-box">
            <div className="result-header">
              <span className="badge">Done</span>
              <span className="file-name">{result.filename}</span>
            </div>

            {/* Cast X-Ray section */}
            {hasCast && (
              <section>
                <h2>Cast X-Ray</h2>

                {result.cast.tmdb_cast?.length > 0 && (
                  <div className="cast-subsection">
                    <p className="cast-source-label">
                      Show cast from TMDB ({result.cast.tmdb_cast.length} members)
                    </p>
                    <div className="cast-grid">
                      {result.cast.tmdb_cast.map((member) => (
                        <div key={member.tmdb_id} className="cast-card">
                          {member.thumbnail ? (
                            <img
                              src={member.thumbnail}
                              alt={member.actor_name}
                              className="cast-photo"
                            />
                          ) : (
                            <div className="cast-photo cast-photo--placeholder">?</div>
                          )}
                          <div className="cast-info">
                            <span className="cast-actor">{member.actor_name}</span>
                            {member.character_name && (
                              <span className="cast-character">as {member.character_name}</span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.cast.detected_faces?.length > 0 && (
                  <div className="cast-subsection">
                    <p className="cast-source-label">
                      Detected faces ({result.cast.detected_faces.length} unique,&nbsp;
                      {result.cast.frames_analysed} frames analysed)
                    </p>
                    <div className="face-grid">
                      {result.cast.detected_faces.map((face) => (
                        <div key={face.face_id} className="face-card">
                          <img
                            src={face.thumbnail}
                            alt={face.actor_name || `Face ${face.face_id}`}
                            className="face-photo"
                          />
                          {face.actor_name ? (
                            <span className="face-actor">{face.actor_name}</span>
                          ) : null}
                          {face.character_name ? (
                            <span className="face-character">as {face.character_name}</span>
                          ) : null}
                          <span className="face-time">{face.screen_time_pct}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            <section>
              <h2>Summary</h2>
              {result.summary
                ? <p className="summary-text">{result.summary}</p>
                : <p className="skipped-notice">Audio extraction and summarisation were disabled via feature flags.</p>
              }
            </section>

            {result.transcript_preview && (
              <section>
                <h2>Transcript Preview</h2>
                <pre className="transcript-preview">{result.transcript_preview}</pre>
              </section>
            )}

            <button className="btn-reset" onClick={reset}>Summarise another video</button>
          </div>
        )}
      </main>
    </div>
  );
}
