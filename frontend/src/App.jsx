import { useState, useRef, useCallback, useEffect } from "react";

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

const DEFAULT_SYSTEM_PROMPT = `\
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
- spoiler-safe summary`;

// ─── MarkdownPreview ──────────────────────────────────────────────────────────
function MarkdownPreview({ text }) {
  const lines = text.split("\n");
  const elements = [];
  let listItems = [];

  const flushList = (key) => {
    if (listItems.length > 0) {
      elements.push(<ul key={`ul-${key}`} className="md-list">{listItems}</ul>);
      listItems = [];
    }
  };

  lines.forEach((line, i) => {
    if (line.startsWith("# ")) {
      flushList(i);
      elements.push(<p key={i} className="md-h1">{line.slice(2)}</p>);
    } else if (line.startsWith("## ")) {
      flushList(i);
      elements.push(<p key={i} className="md-h2">{line.slice(3)}</p>);
    } else if (line.startsWith("- ")) {
      listItems.push(<li key={i}>{line.slice(2)}</li>);
    } else if (line.trim() === "") {
      flushList(i);
      elements.push(<div key={i} className="md-spacer" />);
    } else {
      flushList(i);
      elements.push(<p key={i} className="md-p">{line}</p>);
    }
  });
  flushList("end");

  return <div className="md-content">{elements}</div>;
}

// ─── PromptEditor ─────────────────────────────────────────────────────────────
function PromptEditor({ id, value, onChange }) {
  const [preview, setPreview] = useState(false);
  return (
    <div className="prompt-editor">
      <div className="prompt-editor-tabs">
        <button
          type="button"
          className={`pe-tab ${!preview ? "pe-tab--active" : ""}`}
          onClick={() => setPreview(false)}
        >
          Edit
        </button>
        <button
          type="button"
          className={`pe-tab ${preview ? "pe-tab--active" : ""}`}
          onClick={() => setPreview(true)}
        >
          Preview
        </button>
      </div>
      {preview ? (
        <div className="prompt-preview">
          <MarkdownPreview text={value} />
        </div>
      ) : (
        <textarea
          id={id}
          className="prompt-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
        />
      )}
    </div>
  );
}

const EMPTY_CAST_MEMBER = () => ({ actorName: "", characterName: "", photoFile: null, photoPreview: null });

// ─── ManualCastPanel ──────────────────────────────────────────────────────────
function ManualCastPanel({ selectedShow, setSelectedShow }) {
  const [query, setQuery]           = useState("");
  const [results, setResults]       = useState([]);
  const [searching, setSearching]   = useState(false);
  const [searched, setSearched]     = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [newTitle, setNewTitle]     = useState("");
  const [creating, setCreating]     = useState(false);

  // Add-member form inside selected show
  const [addingMember, setAddingMember]   = useState(false);
  const [newMember, setNewMember]         = useState(EMPTY_CAST_MEMBER());
  const [savingMember, setSavingMember]   = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setSearched(false);
    try {
      const r = await fetch(`/manual-shows/search?q=${encodeURIComponent(query)}`);
      const data = await r.json();
      setResults(data);
    } catch (_) {
      setResults([]);
    } finally {
      setSearching(false);
      setSearched(true);
    }
  };

  const handleSelect = async (show) => {
    const r = await fetch(`/manual-shows/${show.id}`);
    const data = await r.json();
    setSelectedShow(data);
    setResults([]);
    setQuery("");
    setSearched(false);
    setShowCreate(false);
  };

  const handleCreate = async () => {
    if (!newTitle.trim()) return;
    setCreating(true);
    try {
      const fd = new FormData();
      fd.append("title", newTitle.trim());
      const r = await fetch("/manual-shows", { method: "POST", body: fd });
      const data = await r.json();
      // fetch full show (no cast yet)
      const showR = await fetch(`/manual-shows/${data.id}`);
      const showData = await showR.json();
      setSelectedShow(showData);
      setShowCreate(false);
      setNewTitle("");
    } catch (_) {
      alert("Failed to create show");
    } finally {
      setCreating(false);
    }
  };

  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const preview = URL.createObjectURL(file);
    setNewMember((m) => ({ ...m, photoFile: file, photoPreview: preview }));
  };

  const handleAddMember = async () => {
    if (!newMember.actorName.trim() || !selectedShow) return;
    setSavingMember(true);
    try {
      const fd = new FormData();
      fd.append("actor_name", newMember.actorName.trim());
      fd.append("character_name", newMember.characterName.trim());
      if (newMember.photoFile) fd.append("photo", newMember.photoFile);
      const r = await fetch(`/manual-shows/${selectedShow.id}/cast`, { method: "POST", body: fd });
      if (!r.ok) throw new Error("Failed");
      // refresh show
      const showR = await fetch(`/manual-shows/${selectedShow.id}`);
      setSelectedShow(await showR.json());
      setNewMember(EMPTY_CAST_MEMBER());
      setAddingMember(false);
    } catch (_) {
      alert("Failed to add cast member");
    } finally {
      setSavingMember(false);
    }
  };

  const handleDeleteMember = async (castId) => {
    if (!confirm("Delete this cast member?")) return;
    await fetch(`/manual-shows/${selectedShow.id}/cast/${castId}`, { method: "DELETE" });
    const r = await fetch(`/manual-shows/${selectedShow.id}`);
    setSelectedShow(await r.json());
  };

  return (
    <div className="manual-cast-panel">
      {selectedShow ? (
        <div className="manual-selected-show">
          <div className="manual-show-header">
            <span className="manual-show-title">{selectedShow.title}</span>
            <button className="btn-text" onClick={() => setSelectedShow(null)}>Change show</button>
          </div>

          {selectedShow.cast.length === 0 ? (
            <p className="manual-empty">No cast members yet.</p>
          ) : (
            <div className="manual-cast-list">
              {selectedShow.cast.map((m) => (
                <div key={m.id} className="manual-cast-item">
                  {m.photo_data ? (
                    <img src={m.photo_data} alt={m.actor_name} className="manual-cast-thumb" />
                  ) : (
                    <div className="manual-cast-thumb manual-cast-thumb--placeholder">?</div>
                  )}
                  <div className="manual-cast-info">
                    <span className="manual-cast-actor">{m.actor_name}</span>
                    {m.character_name && <span className="manual-cast-char">as {m.character_name}</span>}
                  </div>
                  <button className="btn-icon-danger" onClick={() => handleDeleteMember(m.id)} title="Remove">✕</button>
                </div>
              ))}
            </div>
          )}

          {addingMember ? (
            <div className="manual-add-form">
              <div className="manual-add-row">
                <input
                  placeholder="Actor name *"
                  value={newMember.actorName}
                  onChange={(e) => setNewMember((m) => ({ ...m, actorName: e.target.value }))}
                />
                <input
                  placeholder="Character name"
                  value={newMember.characterName}
                  onChange={(e) => setNewMember((m) => ({ ...m, characterName: e.target.value }))}
                />
              </div>
              <div className="manual-photo-row">
                {newMember.photoPreview && (
                  <img src={newMember.photoPreview} alt="preview" className="manual-cast-thumb" />
                )}
                <label className="btn-upload">
                  {newMember.photoFile ? "Change photo" : "Upload photo"}
                  <input type="file" accept="image/*" onChange={handlePhotoChange} hidden />
                </label>
              </div>
              <div className="manual-add-actions">
                <button className="btn-primary-sm" onClick={handleAddMember} disabled={savingMember}>
                  {savingMember ? "Saving…" : "Save member"}
                </button>
                <button className="btn-text" onClick={() => { setAddingMember(false); setNewMember(EMPTY_CAST_MEMBER()); }}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button className="btn-add-member" onClick={() => setAddingMember(true)}>+ Add cast member</button>
          )}
        </div>
      ) : (
        <>
          <div className="manual-search-row">
            <input
              type="text"
              placeholder="Search show by name…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
            <button className="btn-primary-sm" onClick={handleSearch} disabled={searching}>
              {searching ? "…" : "Search"}
            </button>
          </div>

          {results.length > 0 && (
            <ul className="manual-search-results">
              {results.map((s) => (
                <li key={s.id} onClick={() => handleSelect(s)}>{s.title}</li>
              ))}
            </ul>
          )}

          {searched && results.length === 0 && (
            <div className="manual-not-found">
              <span>No show found.</span>
              <button className="btn-text" onClick={() => { setShowCreate(true); setNewTitle(query); }}>
                Create "{query}"
              </button>
            </div>
          )}

          {showCreate && (
            <div className="manual-create-form">
              <input
                type="text"
                placeholder="Show title"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
              />
              <button className="btn-primary-sm" onClick={handleCreate} disabled={creating}>
                {creating ? "Creating…" : "Create show"}
              </button>
              <button className="btn-text" onClick={() => setShowCreate(false)}>Cancel</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── RelationshipsTab ─────────────────────────────────────────────────────────
function RelationshipsTab({ show }) {
  const [relationships, setRelationships]   = useState([]);
  const [relTypes, setRelTypes]             = useState([]);
  const [newRel, setNewRel]                 = useState({ charAId: "", charBId: "", relType: "" });
  const [showMatrix, setShowMatrix]         = useState(false);
  const [showTypeManager, setShowTypeManager] = useState(false);
  const [newRelType, setNewRelType]         = useState({ name: "", reverseName: "" });

  useEffect(() => {
    fetch("/relationship-types").then((r) => r.json()).then(setRelTypes).catch(() => {});
  }, []);

  useEffect(() => {
    if (!show?.id) { setRelationships([]); return; }
    fetch(`/manual-shows/${show.id}/relationships`).then((r) => r.json()).then(setRelationships).catch(() => setRelationships([]));
  }, [show?.id]);

  const loadRelationships = async () => {
    try {
      const r = await fetch(`/manual-shows/${show.id}/relationships`);
      setRelationships(await r.json());
    } catch (_) { setRelationships([]); }
  };

  const handleAddRelationship = async () => {
    if (!newRel.charAId || !newRel.charBId || !newRel.relType || newRel.charAId === newRel.charBId) return;
    const fd = new FormData();
    fd.append("char_a_id", newRel.charAId);
    fd.append("char_b_id", newRel.charBId);
    fd.append("rel_type_name", newRel.relType);
    await fetch(`/manual-shows/${show.id}/relationships`, { method: "POST", body: fd });
    setNewRel({ charAId: "", charBId: "", relType: "" });
    await loadRelationships();
  };

  const handleDeleteRelationship = async (aId, bId) => {
    await fetch(`/manual-shows/${show.id}/relationships/${aId}/${bId}`, { method: "DELETE" });
    await loadRelationships();
  };

  const handleMatrixChange = async (rowId, colId, value) => {
    if (value === "") {
      await fetch(`/manual-shows/${show.id}/relationships/${rowId}/${colId}`, { method: "DELETE" });
    } else {
      const fd = new FormData();
      fd.append("char_a_id", rowId);
      fd.append("char_b_id", colId);
      fd.append("rel_type_name", value);
      await fetch(`/manual-shows/${show.id}/relationships`, { method: "POST", body: fd });
    }
    await loadRelationships();
  };

  const handleAddRelType = async () => {
    if (!newRelType.name.trim() || !newRelType.reverseName.trim()) return;
    const fd = new FormData();
    fd.append("name", newRelType.name.trim());
    fd.append("reverse_name", newRelType.reverseName.trim());
    const r = await fetch("/relationship-types", { method: "POST", body: fd });
    if (r.ok) {
      const data = await r.json();
      setRelTypes((prev) => [...prev, data].sort((a, b) => a.name.localeCompare(b.name)));
      setNewRelType({ name: "", reverseName: "" });
    }
  };

  const cast = show?.cast ?? [];

  // Group relationships by char_a_id
  const relsByChar = {};
  for (const rel of relationships) {
    if (!relsByChar[rel.char_a_id]) relsByChar[rel.char_a_id] = [];
    relsByChar[rel.char_a_id].push(rel);
  }

  const FAMILY_TYPES = new Set(["Father", "Mother", "Child", "Parent", "Brother", "Sister"]);
  const ROMANTIC_TYPES = new Set(["Lover", "Spouse"]);

  const getCellClass = (relType) => {
    if (!relType) return "matrix-cell";
    if (FAMILY_TYPES.has(relType)) return "matrix-cell matrix-cell--family";
    if (ROMANTIC_TYPES.has(relType)) return "matrix-cell matrix-cell--romantic";
    return "matrix-cell";
  };

  return (
    <div className="rel-tab-content">
      {/* Add relationship row */}
      <div className="rel-section-label">Add Relationship</div>
      <div className="rel-add-form">
        <select value={newRel.charAId} onChange={(e) => setNewRel((p) => ({ ...p, charAId: e.target.value }))}>
          <option value="">Character A</option>
          {cast.map((m) => (
            <option key={m.id} value={m.id}>{m.character_name || m.actor_name}</option>
          ))}
        </select>
        <span className="rel-is">is</span>
        <select value={newRel.relType} onChange={(e) => setNewRel((p) => ({ ...p, relType: e.target.value }))}>
          <option value="">— type —</option>
          {relTypes.map((t) => (
            <option key={t.id} value={t.name}>{t.name}</option>
          ))}
        </select>
        <span className="rel-is">of</span>
        <select value={newRel.charBId} onChange={(e) => setNewRel((p) => ({ ...p, charBId: e.target.value }))}>
          <option value="">Character B</option>
          {cast.map((m) => (
            <option key={m.id} value={m.id}>{m.character_name || m.actor_name}</option>
          ))}
        </select>
        <button className="btn-primary-sm" onClick={handleAddRelationship}>+</button>
      </div>

      {/* Character cards */}
      {cast.length > 0 && (
        <div className="rel-cards">
          {cast.map((m) => {
            const rels = relsByChar[m.id];
            if (!rels || rels.length === 0) return null;
            return (
              <div key={m.id} className="rel-card">
                <div className="rel-card-header">
                  {m.photo_data
                    ? <img src={m.photo_data} alt={m.actor_name} className="rel-card-thumb" />
                    : <div className="rel-card-thumb rel-card-thumb--placeholder">?</div>}
                  <span className="rel-card-name">
                    {m.character_name || m.actor_name}
                    {m.character_name && <span className="rel-card-actor"> ({m.actor_name})</span>}
                  </span>
                </div>
                <ul className="rel-list">
                  {rels.map((rel) => (
                    <li key={rel.id} className="rel-item">
                      <span>{rel.rel_type} of {rel.char_b_name || rel.char_b_actor}</span>
                      <button
                        className="btn-icon-danger"
                        onClick={() => handleDeleteRelationship(rel.char_a_id, rel.char_b_id)}
                      >✕</button>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
          {cast.every((m) => !relsByChar[m.id] || relsByChar[m.id].length === 0) && (
            <p className="manual-empty">No relationships yet. Add one above.</p>
          )}
        </div>
      )}

      {/* Matrix panel */}
      <div className="matrix-panel">
        <button className="matrix-panel-toggle" onClick={() => setShowMatrix((v) => !v)}>
          Advanced: Relationship Matrix {showMatrix ? "▲ Hide" : "▼ Show"}
        </button>
        {showMatrix && cast.length > 0 && (
          <div className="matrix-scroll">
            <table className="matrix-table">
              <thead>
                <tr>
                  <th className="matrix-th"></th>
                  {cast.map((m) => (
                    <th key={m.id} className="matrix-th">{m.character_name || m.actor_name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cast.map((rowM) => (
                  <tr key={rowM.id}>
                    <th className="matrix-th">{rowM.character_name || rowM.actor_name}</th>
                    {cast.map((colM) => {
                      if (rowM.id === colM.id) {
                        return <td key={colM.id} className="matrix-cell matrix-cell--self">—</td>;
                      }
                      const existing = relationships.find(
                        (r) => r.char_a_id === rowM.id && r.char_b_id === colM.id
                      );
                      const val = existing?.rel_type ?? "";
                      return (
                        <td key={colM.id} className={getCellClass(val)}>
                          <select
                            value={val}
                            onChange={(e) => handleMatrixChange(rowM.id, colM.id, e.target.value)}
                          >
                            <option value="">—</option>
                            {relTypes.map((t) => (
                              <option key={t.id} value={t.name}>{t.name}</option>
                            ))}
                          </select>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Manage relationship types */}
      <div className="type-manager">
        <button className="matrix-panel-toggle" onClick={() => setShowTypeManager((v) => !v)}>
          Manage Relationship Types {showTypeManager ? "▲ Hide" : "▼ Show"}
        </button>
        {showTypeManager && (
          <div className="type-manager-body">
            {relTypes.map((t) => (
              <div key={t.id} className="type-row">
                <span className="type-name">{t.name}</span>
                <span className="type-arrow">↔</span>
                <span className="type-reverse">{t.reverse_name}</span>
              </div>
            ))}
            <div className="type-add-row">
              <input
                placeholder="Type name"
                value={newRelType.name}
                onChange={(e) => setNewRelType((p) => ({ ...p, name: e.target.value }))}
              />
              <span className="type-arrow">↔</span>
              <input
                placeholder="Reverse name"
                value={newRelType.reverseName}
                onChange={(e) => setNewRelType((p) => ({ ...p, reverseName: e.target.value }))}
              />
              <button className="btn-primary-sm" onClick={handleAddRelType}>+ Add</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── CastSavePanel ────────────────────────────────────────────────────────────
function CastSavePanel({ castMembers }) {
  const [open, setOpen]         = useState(false);
  const [shows, setShows]       = useState([]);
  const [showId, setShowId]     = useState("");
  const [dbShow, setDbShow]     = useState(null);
  const [saving, setSaving]     = useState({});
  const [editingId, setEditingId] = useState(null);
  const [editFields, setEditFields] = useState({});

  useEffect(() => {
    fetch("/manual-shows").then((r) => r.json()).then(setShows).catch(() => {});
  }, []);

  const loadDbShow = async (id) => {
    const r = await fetch(`/manual-shows/${id}`);
    setDbShow(await r.json());
  };

  const handleSelectShow = async (id) => {
    setShowId(id);
    setEditingId(null);
    if (id) await loadDbShow(Number(id));
    else setDbShow(null);
  };

  const findInDb = (member) =>
    dbShow?.cast.find(
      (m) => m.actor_name.toLowerCase() === (member.actor_name || "").toLowerCase()
    ) ?? null;

  const handleAdd = async (member) => {
    const key = member.actor_name;
    setSaving((p) => ({ ...p, [key]: true }));
    try {
      const fd = new FormData();
      fd.append("actor_name", member.actor_name || "Unknown");
      fd.append("character_name", member.character_name || "");
      if (member.thumbnail) fd.append("photo_url", member.thumbnail);
      const r = await fetch(`/manual-shows/${dbShow.id}/cast`, { method: "POST", body: fd });
      if (r.ok) await loadDbShow(dbShow.id);
    } finally {
      setSaving((p) => ({ ...p, [key]: false }));
    }
  };

  const startEdit = (dbMember) => {
    setEditingId(dbMember.id);
    setEditFields({ actorName: dbMember.actor_name, characterName: dbMember.character_name || "" });
  };

  const handleEditSave = async () => {
    const fd = new FormData();
    fd.append("actor_name", editFields.actorName);
    fd.append("character_name", editFields.characterName);
    await fetch(`/manual-shows/${dbShow.id}/cast/${editingId}`, { method: "PUT", body: fd });
    setEditingId(null);
    await loadDbShow(dbShow.id);
  };

  if (!castMembers || castMembers.length === 0) return null;

  return (
    <div className="cast-save-panel">
      <button className="cast-save-toggle" onClick={() => setOpen((v) => !v)}>
        Save cast to database
        <span className="chevron" style={{ marginLeft: 6 }}>{open ? "›" : "›"}</span>
      </button>

      {open && (
        <div className="cast-save-body">
          <div className="cast-save-show-row">
            <label>Show:</label>
            <select value={showId} onChange={(e) => handleSelectShow(e.target.value)}>
              <option value="">— Select a show —</option>
              {shows.map((s) => (
                <option key={s.id} value={s.id}>{s.title}</option>
              ))}
            </select>
          </div>

          {dbShow && (
            <div className="cast-save-list">
              {castMembers.map((member, i) => {
                const dbMember = findInDb(member);
                const key = member.actor_name || `face-${i}`;
                const isSaving = saving[key];

                if (editingId === dbMember?.id) {
                  return (
                    <div key={key} className="cast-save-item cast-save-item--editing">
                      <div className="cast-save-edit-fields">
                        <input
                          value={editFields.actorName}
                          onChange={(e) => setEditFields((p) => ({ ...p, actorName: e.target.value }))}
                          placeholder="Actor name"
                        />
                        <input
                          value={editFields.characterName}
                          onChange={(e) => setEditFields((p) => ({ ...p, characterName: e.target.value }))}
                          placeholder="Character name"
                        />
                      </div>
                      <div className="cast-save-actions">
                        <button className="btn-primary-sm" onClick={handleEditSave}>Save</button>
                        <button className="btn-text" onClick={() => setEditingId(null)}>Cancel</button>
                      </div>
                    </div>
                  );
                }

                return (
                  <div key={key} className="cast-save-item">
                    {member.thumbnail ? (
                      <img src={member.thumbnail} alt={member.actor_name} className="cast-save-thumb" />
                    ) : (
                      <div className="cast-save-thumb cast-save-thumb--placeholder">?</div>
                    )}
                    <div className="cast-save-info">
                      <span className="cast-save-actor">{member.actor_name || "Unknown"}</span>
                      {member.character_name && (
                        <span className="cast-save-char">as {member.character_name}</span>
                      )}
                    </div>
                    {dbMember ? (
                      <div className="cast-save-added">
                        <span className="cast-save-status">✓ Added</span>
                        <button className="btn-text" onClick={() => startEdit(dbMember)}>Edit</button>
                      </div>
                    ) : (
                      <button
                        className="btn-primary-sm"
                        onClick={() => handleAdd(member)}
                        disabled={isSaving}
                      >
                        {isSaving ? "Adding…" : "+ Add"}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── CastAdminModal ───────────────────────────────────────────────────────────
function CastAdminModal({ onClose }) {
  const [adminTab, setAdminTab]           = useState("cast");
  const [shows, setShows]                 = useState([]);
  const [loading, setLoading]             = useState(true);
  const [selectedShowId, setSelectedShowId] = useState("");
  const [selectedShow, setSelectedShow]   = useState(null);
  const [editingId, setEditingId]         = useState(null);
  const [editFields, setEditFields]       = useState({});
  const [addingMember, setAddingMember]   = useState(false);
  const [newMember, setNewMember]         = useState(EMPTY_CAST_MEMBER());
  const [savingMember, setSavingMember]   = useState(false);

  const loadShows = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch("/manual-shows");
      setShows(await r.json());
    } finally {
      setLoading(false);
    }
  }, []);

  useState(() => { loadShows(); }, []);

  const refreshSelectedShow = async (id) => {
    const r = await fetch(`/manual-shows/${id}`);
    setSelectedShow(await r.json());
  };

  const handleSelectShow = async (id) => {
    setSelectedShowId(id);
    setEditingId(null);
    setAddingMember(false);
    setNewMember(EMPTY_CAST_MEMBER());
    if (!id) { setSelectedShow(null); return; }
    await refreshSelectedShow(Number(id));
  };

  const handleDeleteShow = async () => {
    if (!selectedShow) return;
    if (!confirm("Delete this show and all its cast members?")) return;
    await fetch(`/manual-shows/${selectedShow.id}`, { method: "DELETE" });
    setShows((prev) => prev.filter((s) => s.id !== selectedShow.id));
    setSelectedShowId("");
    setSelectedShow(null);
  };

  const handleDeleteMember = async (castId) => {
    if (!confirm("Delete this cast member?")) return;
    await fetch(`/manual-shows/${selectedShow.id}/cast/${castId}`, { method: "DELETE" });
    await refreshSelectedShow(selectedShow.id);
  };

  const startEdit = (m) => {
    setEditingId(m.id);
    setEditFields({ actorName: m.actor_name, characterName: m.character_name || "", photoFile: null, photoPreview: m.photo_data });
  };

  const handleEditSave = async (castId) => {
    const fd = new FormData();
    fd.append("actor_name", editFields.actorName);
    fd.append("character_name", editFields.characterName);
    if (editFields.photoFile) fd.append("photo", editFields.photoFile);
    await fetch(`/manual-shows/${selectedShow.id}/cast/${castId}`, { method: "PUT", body: fd });
    setEditingId(null);
    await refreshSelectedShow(selectedShow.id);
  };

  const handleAddMember = async () => {
    if (!newMember.actorName.trim()) return;
    setSavingMember(true);
    try {
      const fd = new FormData();
      fd.append("actor_name", newMember.actorName.trim());
      fd.append("character_name", newMember.characterName.trim());
      if (newMember.photoFile) fd.append("photo", newMember.photoFile);
      await fetch(`/manual-shows/${selectedShow.id}/cast`, { method: "POST", body: fd });
      setNewMember(EMPTY_CAST_MEMBER());
      setAddingMember(false);
      await refreshSelectedShow(selectedShow.id);
    } finally {
      setSavingMember(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <div className="modal-header">
          <h2 style={{ textTransform: "none", fontSize: "1rem", color: "#e8e8f0" }}>Cast Database</h2>
          <button className="btn-icon" onClick={onClose}>✕</button>
        </div>

        {/* Unified show selector */}
        <div className="admin-show-selector-row">
          <select value={selectedShowId} onChange={(e) => handleSelectShow(e.target.value)} disabled={loading}>
            <option value="">— Select a show —</option>
            {shows.map((s) => (
              <option key={s.id} value={s.id}>{s.title}</option>
            ))}
          </select>
          {selectedShow && (
            <button className="btn-icon-danger" onClick={handleDeleteShow}>Delete show</button>
          )}
        </div>

        {!selectedShow ? (
          loading ? (
            <div className="modal-empty">Loading…</div>
          ) : shows.length === 0 ? (
            <div className="modal-empty">No manual shows yet. Add one via settings.</div>
          ) : (
            <div className="modal-empty">Select a show above to manage its cast and relationships.</div>
          )
        ) : (
          <>
            <div className="admin-tabs">
              <button
                className={`admin-tab ${adminTab === "cast" ? "admin-tab--active" : ""}`}
                onClick={() => setAdminTab("cast")}
              >Cast Members</button>
              <button
                className={`admin-tab ${adminTab === "relationships" ? "admin-tab--active" : ""}`}
                onClick={() => setAdminTab("relationships")}
              >Relationships</button>
            </div>

            {adminTab === "cast" ? (
              <div className="admin-shows-list">
                <div className="admin-cast-list">
                  {selectedShow.cast.length === 0 && !addingMember && (
                    <p className="manual-empty">No cast members yet.</p>
                  )}
                  {selectedShow.cast.map((m) => (
                    <div key={m.id} className="admin-cast-row">
                      {editingId === m.id ? (
                        <div className="admin-edit-form">
                          <div className="manual-photo-row">
                            {editFields.photoPreview && (
                              <img src={editFields.photoPreview} alt="" className="manual-cast-thumb" />
                            )}
                            <label className="btn-upload">
                              Change photo
                              <input type="file" accept="image/*" hidden onChange={(e) => {
                                const f = e.target.files[0];
                                if (f) setEditFields((p) => ({ ...p, photoFile: f, photoPreview: URL.createObjectURL(f) }));
                              }} />
                            </label>
                          </div>
                          <input value={editFields.actorName} onChange={(e) => setEditFields((p) => ({ ...p, actorName: e.target.value }))} placeholder="Actor name" />
                          <input value={editFields.characterName} onChange={(e) => setEditFields((p) => ({ ...p, characterName: e.target.value }))} placeholder="Character name" />
                          <div className="manual-add-actions">
                            <button className="btn-primary-sm" onClick={() => handleEditSave(m.id)}>Save</button>
                            <button className="btn-text" onClick={() => setEditingId(null)}>Cancel</button>
                          </div>
                        </div>
                      ) : (
                        <>
                          {m.photo_data
                            ? <img src={m.photo_data} alt={m.actor_name} className="manual-cast-thumb" />
                            : <div className="manual-cast-thumb manual-cast-thumb--placeholder">?</div>}
                          <div className="manual-cast-info">
                            <span className="manual-cast-actor">{m.actor_name}</span>
                            {m.character_name && <span className="manual-cast-char">as {m.character_name}</span>}
                          </div>
                          <button className="btn-text" onClick={() => startEdit(m)}>Edit</button>
                          <button className="btn-icon-danger" onClick={() => handleDeleteMember(m.id)}>✕</button>
                        </>
                      )}
                    </div>
                  ))}
                </div>

                {addingMember ? (
                  <div className="manual-add-form" style={{ margin: "0 12px 12px" }}>
                    <div className="manual-add-row">
                      <input placeholder="Actor name *" value={newMember.actorName} onChange={(e) => setNewMember((m) => ({ ...m, actorName: e.target.value }))} />
                      <input placeholder="Character name" value={newMember.characterName} onChange={(e) => setNewMember((m) => ({ ...m, characterName: e.target.value }))} />
                    </div>
                    <div className="manual-photo-row">
                      {newMember.photoPreview && <img src={newMember.photoPreview} alt="preview" className="manual-cast-thumb" />}
                      <label className="btn-upload">
                        {newMember.photoFile ? "Change photo" : "Upload photo"}
                        <input type="file" accept="image/*" onChange={(e) => {
                          const f = e.target.files[0];
                          if (f) setNewMember((m) => ({ ...m, photoFile: f, photoPreview: URL.createObjectURL(f) }));
                        }} hidden />
                      </label>
                    </div>
                    <div className="manual-add-actions">
                      <button className="btn-primary-sm" onClick={handleAddMember} disabled={savingMember}>
                        {savingMember ? "Saving…" : "Save member"}
                      </button>
                      <button className="btn-text" onClick={() => { setAddingMember(false); setNewMember(EMPTY_CAST_MEMBER()); }}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: "0 12px 12px" }}>
                    <button className="btn-add-member" onClick={() => setAddingMember(true)}>+ Add cast member</button>
                  </div>
                )}
              </div>
            ) : (
              <div className="admin-shows-list">
                <RelationshipsTab show={selectedShow} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
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

  // Cast source
  const [castSource, setCastSource]         = useState("tmdb");
  const [selectedManualShow, setSelectedManualShow] = useState(null);

  // Feature flags
  const [featureFlags, setFeatureFlags] = useState(DEFAULT_FEATURE_FLAGS);
  const toggleFlag = (key) =>
    setFeatureFlags((prev) => ({ ...prev, [key]: !prev[key] }));

  // Cache state
  const [isCacheHit, setIsCacheHit] = useState(false);
  const [pendingFile, setPendingFile] = useState(null);

  // Regenerate summary state
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [regenPrompt, setRegenPrompt]       = useState(DEFAULT_SYSTEM_PROMPT);
  const [regenModel, setRegenModel]         = useState("gpt-4o-mini");
  const [regenMaxWords, setRegenMaxWords]   = useState(200);
  const [regenMaxTokens, setRegenMaxTokens] = useState(500);

  // Admin modal
  const [showAdminModal, setShowAdminModal] = useState(false);

  const uploadAndProcess = async (file) => {
    setStage("uploading");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);
    formData.append("system_prompt", systemPrompt);
    formData.append("max_tokens", String(maxTokens));
    formData.append("max_words", String(maxWords));
    formData.append("show_title", showTitle);
    formData.append("features", JSON.stringify(featureFlags));
    formData.append("cast_source", castSource);
    formData.append("manual_show_id", String(selectedManualShow?.id ?? 0));

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

      const response = await fetch("/summarise", { method: "POST", body: formData });
      clearInterval(labelInterval);

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Processing failed");
      }

      const data = await response.json();
      setResult(data);
      setRegenPrompt(data.system_prompt || DEFAULT_SYSTEM_PROMPT);
      setRegenModel(data.model || "gpt-4o-mini");
      setRegenMaxWords(data.max_words || 200);
      setStage("done");
    } catch (err) {
      setError(err.message);
      setStage("error");
    }
  };

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
    setIsCacheHit(false);
    setPendingFile(file);

    // Cache pre-check
    try {
      const cacheResp = await fetch(`/check-cache?filename=${encodeURIComponent(file.name)}`);
      if (cacheResp.ok) {
        const cacheData = await cacheResp.json();
        if (cacheData.cached && cacheData.result) {
          setResult(cacheData.result);
          setRegenPrompt(cacheData.result.system_prompt || DEFAULT_SYSTEM_PROMPT);
          setRegenModel(cacheData.result.model || "gpt-4o-mini");
          setRegenMaxWords(cacheData.result.max_words || 200);
          setIsCacheHit(true);
          setStage("done");
          return;
        }
      }
    } catch (_) { /* non-fatal */ }

    await uploadAndProcess(file);
  };

  const handleProcessFresh = async () => {
    if (!pendingFile) return;
    setIsCacheHit(false);
    setResult(null);
    setError(null);
    await uploadAndProcess(pendingFile);
  };

  const handleRegenerate = async () => {
    if (!result?.filename) return;
    setIsRegenerating(true);
    try {
      const formData = new FormData();
      formData.append("filename", result.filename);
      formData.append("model", regenModel);
      formData.append("system_prompt", regenPrompt);
      formData.append("max_tokens", String(regenMaxTokens));
      formData.append("max_words", String(regenMaxWords));

      const resp = await fetch("/regenerate-summary", { method: "POST", body: formData });
      if (!resp.ok) {
        const d = await resp.json();
        throw new Error(d.detail || "Regeneration failed");
      }
      const d = await resp.json();
      setResult((prev) => ({ ...prev, summary: d.summary, system_prompt: d.system_prompt }));
    } catch (err) {
      alert(`Regeneration failed: ${err.message}`);
    } finally {
      setIsRegenerating(false);
    }
  };

  const onInputChange = (e) => { handleFile(e.target.files[0]); e.target.value = ""; };
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); };
  const reset  = () => {
    setStage("idle"); setResult(null); setError(null); setFileName(null);
    setIsCacheHit(false); setPendingFile(null);
  };

  const isProcessing = Object.keys(STAGES)
    .filter((k) => STAGES[k] !== null && k !== "done" && k !== "error")
    .includes(stage);

  const hasCast = result?.cast && (
    result.cast.detected_faces?.length > 0 || result.cast.tmdb_cast?.length > 0
  );

  const castSectionLabel =
    result?.cast?.cast_source === "manual"      ? "Manual cast" :
    result?.cast?.cast_source === "tmdb_local"  ? "Show cast (from local DB — synced from TMDB)" :
                                                  "Show cast from TMDB";

  return (
    <div className="container">
      <header>
        <div className="header-row">
          <div>
            <h1>Video Summariser</h1>
            <p className="subtitle">Upload a video to extract audio, analyse cast, and generate an AI summary</p>
          </div>
          <button className="btn-admin" onClick={() => setShowAdminModal(true)}>Cast Database</button>
        </div>
      </header>

      {showAdminModal && <CastAdminModal onClose={() => setShowAdminModal(false)} />}

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
                  {/* Cast source toggle */}
                  <div className="setting-row setting-row--full">
                    <label>Cast lookup source</label>
                    <div className="cast-source-toggle">
                      <label className="radio-label">
                        <input
                          type="radio"
                          value="tmdb"
                          checked={castSource === "tmdb"}
                          onChange={() => setCastSource("tmdb")}
                        />
                        TMDB (automatic)
                      </label>
                      <label className="radio-label">
                        <input
                          type="radio"
                          value="manual"
                          checked={castSource === "manual"}
                          onChange={() => setCastSource("manual")}
                        />
                        Manual database
                      </label>
                    </div>
                  </div>

                  {castSource === "tmdb" && (
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
                  )}

                  {castSource === "manual" && (
                    <div className="setting-row setting-row--full">
                      <label>Manual cast</label>
                      <ManualCastPanel
                        selectedShow={selectedManualShow}
                        setSelectedShow={setSelectedManualShow}
                      />
                    </div>
                  )}

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
                    <label>System prompt</label>
                    <PromptEditor
                      id="system-prompt"
                      value={systemPrompt}
                      onChange={setSystemPrompt}
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

            {/* Cache hit banner */}
            {isCacheHit && (
              <div className="cache-banner">
                <span>⚡ Results loaded from cache</span>
                <button className="btn-process-fresh" onClick={handleProcessFresh}>
                  Process fresh →
                </button>
              </div>
            )}

            {/* Cast X-Ray section */}
            {hasCast && (
              <section>
                <h2>Cast X-Ray</h2>

                {result.cast.tmdb_cast?.length > 0 && (
                  <div className="cast-subsection">
                    <p className="cast-source-label">
                      {castSectionLabel} ({result.cast.tmdb_cast.length} members)
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

                <CastSavePanel
                  castMembers={[
                    ...(result.cast.tmdb_cast || []).map((m) => ({
                      actor_name: m.actor_name,
                      character_name: m.character_name,
                      thumbnail: m.thumbnail,
                    })),
                    ...(result.cast.detected_faces || [])
                      .filter((f) => f.actor_name)
                      .map((f) => ({
                        actor_name: f.actor_name,
                        character_name: f.character_name,
                        thumbnail: f.thumbnail,
                      })),
                  ]}
                />
              </section>
            )}

            {/* Regenerate Summary section */}
            {result.summary && (
              <section className="regen-section">
                <h2>Regenerate Summary</h2>
                <div className="setting-row setting-row--full">
                  <label>System prompt</label>
                  <PromptEditor
                    id="regen-prompt"
                    value={regenPrompt}
                    onChange={setRegenPrompt}
                  />
                </div>
                <div className="regen-controls">
                  <div className="setting-row">
                    <label htmlFor="regen-model">Model</label>
                    <select
                      id="regen-model"
                      value={regenModel}
                      onChange={(e) => setRegenModel(e.target.value)}
                    >
                      {MODELS.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                  <div className="setting-row">
                    <label htmlFor="regen-max-words">Max words</label>
                    <input
                      id="regen-max-words"
                      type="number"
                      min={50} max={1000} step={50}
                      value={regenMaxWords}
                      onChange={(e) => setRegenMaxWords(Number(e.target.value))}
                    />
                  </div>
                </div>
                <button
                  className="btn-regen"
                  onClick={handleRegenerate}
                  disabled={isRegenerating}
                >
                  {isRegenerating ? (
                    <><span className="spinner-inline" /> Regenerating…</>
                  ) : (
                    "Regenerate Summary ↻"
                  )}
                </button>
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
