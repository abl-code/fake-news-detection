import { useState, useRef, useEffect } from "react";

const API = "http://localhost:5000";

const SAMPLE_URLS = [
  "https://www.bbc.com/news/world",
  "https://timesofindia.indiatimes.com/",
  "https://theonion.com/",
];

const MODEL_DESCRIPTIONS = {
  "Logistic Regression": "Best overall — strong on full-length articles",
  "Linear SVM":          "Fast, close accuracy to LR",
  "Gradient Boosting":   "Highest nuance, slower inference",
  "Random Forest":       "Robust ensemble, moderate speed",
  "Naive Bayes":         "Baseline — fastest, good for headlines",
};

function ConfidenceRing({ value, label, color }) {
  const r = 36, circ = 2 * Math.PI * r;
  const dash = (value / 100) * circ;
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:6 }}>
      <svg width="90" height="90" viewBox="0 0 90 90" role="img" aria-label={`${label}: ${value}%`}>
        <circle cx="45" cy="45" r={r} fill="none" stroke="#e5e7eb" strokeWidth="7"/>
        <circle cx="45" cy="45" r={r} fill="none" stroke={color} strokeWidth="7"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 45 45)"
          style={{ transition:"stroke-dasharray 0.9s cubic-bezier(.4,0,.2,1)" }}/>
        <text x="45" y="49" textAnchor="middle" fontSize="14" fontWeight="600"
          fontFamily="'DM Mono',monospace" fill={color}>{value}%</text>
      </svg>
      <span style={{ fontSize:11, color:"#6b7280", letterSpacing:"0.08em", textTransform:"uppercase" }}>{label}</span>
    </div>
  );
}

function Pill({ label, color, bg }) {
  return (
    <span style={{ display:"inline-flex", alignItems:"center", gap:6, background:bg, color,
      borderRadius:100, padding:"4px 14px", fontSize:12, fontWeight:600,
      letterSpacing:"0.05em", textTransform:"uppercase" }}>
      <span style={{ width:7, height:7, borderRadius:"50%", background:color }}/>
      {label}
    </span>
  );
}

function Spinner() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none"
      style={{ animation:"spin 0.8s linear infinite" }} aria-hidden="true">
      <circle cx="10" cy="10" r="8" stroke="#d1d5db" strokeWidth="2.5"/>
      <path d="M10 2a8 8 0 018 8" stroke="#111" strokeWidth="2.5" strokeLinecap="round"/>
    </svg>
  );
}

export default function FakeNewsDetector() {
  const [mode, setMode]             = useState("url");      // "url" | "text"
  const [url, setUrl]               = useState("");
  const [articleText, setArticleText] = useState("");
  const [articleTitle, setArticleTitle] = useState("");
  const [selectedModel, setSelectedModel] = useState("Logistic Regression");
  const [availableModels, setAvailableModels] = useState([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [status, setStatus]         = useState("idle");
  const [result, setResult]         = useState(null);
  const [errorMsg, setErrorMsg]     = useState("");
  const [showPreview, setShowPreview] = useState(false);
  const inputRef  = useRef(null);
  const menuRef   = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, [mode]);

  useEffect(() => {
    fetch(`${API}/models`)
      .then(r => r.json())
      .then(data => {
        if (data.models?.length) {
          setAvailableModels(data.models);
          setSelectedModel(data.default || data.models[0]);
        }
      })
      .catch(() => {
        setAvailableModels(Object.keys(MODEL_DESCRIPTIONS));
      });
  }, []);

  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target))
        setShowModelMenu(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  async function analyze() {
    setStatus("loading");
    setResult(null);
    setErrorMsg("");
    setShowPreview(false);

    try {
      let res, data;

      if (mode === "url") {
        const trimmed = url.trim();
        if (!trimmed) { setStatus("idle"); return; }
        if (!/^https?:\/\//i.test(trimmed)) {
          setErrorMsg("URL must start with http:// or https://");
          setStatus("error"); return;
        }
        res = await fetch(`${API}/predict`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({ url: trimmed, model: selectedModel }),
        });
      } else {
        const trimmed = articleText.trim();
        if (trimmed.split(/\s+/).length < 10) {
          setErrorMsg("Please paste at least 10 words of article text.");
          setStatus("error"); return;
        }
        res = await fetch(`${API}/predict-text`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({ text: trimmed, title: articleTitle.trim(), model: selectedModel }),
        });
      }

      data = await res.json();
      if (!res.ok || data.error) {
        setErrorMsg(data.error || "Something went wrong.");
        setStatus("error");
      } else {
        setResult(data);
        setStatus("result");
      }
    } catch {
      setErrorMsg("Could not reach the backend. Make sure app.py is running on port 5000.");
      setStatus("error");
    }
  }

  function reset() {
    setUrl(""); setArticleText(""); setArticleTitle("");
    setStatus("idle"); setResult(null); setErrorMsg(""); setShowPreview(false);
    setTimeout(() => inputRef.current?.focus(), 50);
  }

  const isReal  = result?.label === "REAL";
  const verdict = isReal
    ? { word:"Real",  color:"#059669", bg:"#ecfdf5", ring:"#10b981" }
    : { word:"Fake",  color:"#dc2626", bg:"#fef2f2", ring:"#ef4444" };

  const canSubmit = status !== "loading" &&
    (mode === "url" ? url.trim() : articleText.trim().split(/\s+/).length >= 10);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');
        @keyframes spin { to { transform:rotate(360deg); } }
        @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
        * { box-sizing:border-box; }
        body { margin:0; background:#f9f7f4; }

        .mode-tab {
          flex:1; padding:9px 0; font-size:13px; font-family:'DM Sans',sans-serif;
          background:none; border:none; cursor:pointer; border-radius:8px;
          transition:background .15s, color .15s; color:#6b7280; font-weight:500;
        }
        .mode-tab.active { background:#fff; color:#111; box-shadow:0 1px 3px rgba(0,0,0,0.08); }

        .url-input {
          width:100%; border:1.5px solid #d1d5db; border-radius:10px;
          padding:13px 16px; font-size:14px; font-family:'DM Mono',monospace;
          background:#fff; color:#111; outline:none;
          transition:border-color .2s, box-shadow .2s;
        }
        .url-input:focus { border-color:#111; box-shadow:0 0 0 3px rgba(17,17,17,0.07); }
        .url-input::placeholder { color:#9ca3af; }

        .text-input {
          width:100%; border:1.5px solid #d1d5db; border-radius:10px;
          padding:13px 16px; font-size:14px; font-family:'DM Sans',sans-serif;
          background:#fff; color:#111; outline:none; resize:vertical; min-height:140px;
          transition:border-color .2s, box-shadow .2s; line-height:1.6;
        }
        .text-input:focus { border-color:#111; box-shadow:0 0 0 3px rgba(17,17,17,0.07); }
        .text-input::placeholder { color:#9ca3af; }

        .title-input {
          width:100%; border:1.5px solid #e5e7eb; border-radius:8px;
          padding:10px 14px; font-size:13px; font-family:'DM Sans',sans-serif;
          background:#fff; color:#111; outline:none;
          transition:border-color .2s;
        }
        .title-input:focus { border-color:#111; }
        .title-input::placeholder { color:#9ca3af; }

        .analyze-btn {
          background:#111; color:#fff; border:none; border-radius:10px;
          padding:13px 28px; font-size:14px; font-weight:500;
          font-family:'DM Sans',sans-serif; cursor:pointer; white-space:nowrap;
          transition:background .15s, transform .1s;
          display:flex; align-items:center; gap:8px;
        }
        .analyze-btn:hover:not(:disabled) { background:#333; }
        .analyze-btn:active:not(:disabled) { transform:scale(0.97); }
        .analyze-btn:disabled { opacity:0.45; cursor:not-allowed; }

        .model-btn {
          display:flex; align-items:center; gap:8px;
          background:#fff; border:1.5px solid #e5e7eb; border-radius:10px;
          padding:9px 14px; font-size:13px; font-family:'DM Mono',monospace;
          cursor:pointer; color:#111; transition:border-color .15s;
          white-space:nowrap;
        }
        .model-btn:hover { border-color:#9ca3af; }

        .model-menu {
          position:absolute; top:calc(100% + 6px); left:0; right:0; z-index:100;
          background:#fff; border:1.5px solid #e5e7eb; border-radius:12px;
          box-shadow:0 8px 24px rgba(0,0,0,0.1); overflow:hidden;
        }
        .model-option {
          width:100%; text-align:left; background:none; border:none; cursor:pointer;
          padding:10px 14px; font-family:'DM Sans',sans-serif; font-size:13px;
          color:#374151; transition:background .1s;
        }
        .model-option:hover { background:#f9f7f4; }
        .model-option.selected { background:#f3f4f6; color:#111; font-weight:500; }
        .model-option-desc { font-size:11px; color:#9ca3af; display:block; margin-top:1px; }

        .reset-btn {
          background:none; border:1.5px solid #d1d5db; border-radius:8px;
          padding:7px 16px; font-size:13px; cursor:pointer; color:#6b7280;
          font-family:'DM Sans',sans-serif; transition:border-color .15s, color .15s;
        }
        .reset-btn:hover { border-color:#9ca3af; color:#374151; }

        .result-card {
          background:#fff; border:1.5px solid #e5e7eb; border-radius:16px;
          padding:28px; animation:fadeUp .4s ease;
        }
        .meta-chip {
          display:inline-flex; align-items:center; gap:5px;
          background:#f3f4f6; border-radius:6px; padding:4px 10px;
          font-size:12px; color:#4b5563; font-family:'DM Mono',monospace;
        }
        .preview-text {
          font-size:13px; line-height:1.7; color:#4b5563;
          background:#f9f7f4; border-radius:8px; padding:14px 16px;
          border-left:3px solid #d1d5db; margin-top:12px;
          font-family:'DM Sans',sans-serif;
        }
        .sample-btn {
          background:none; border:1px solid #e5e7eb; border-radius:6px;
          padding:4px 12px; font-size:12px; cursor:pointer; color:#6b7280;
          font-family:'DM Mono',monospace; transition:all .15s;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:220px;
        }
        .sample-btn:hover { border-color:#9ca3af; color:#111; background:#f9f7f4; }
        .toggle-preview {
          background:none; border:none; font-size:12px; color:#6b7280;
          cursor:pointer; font-family:'DM Sans',sans-serif; padding:0;
          text-decoration:underline; text-underline-offset:2px;
        }
        .toggle-preview:hover { color:#111; }
        .word-count {
          font-size:11px; color:#9ca3af; text-align:right; margin-top:4px;
          font-family:'DM Mono',monospace;
        }
      `}</style>

      <div style={{ minHeight:"100vh", background:"#f9f7f4", padding:"40px 16px", fontFamily:"'DM Sans',sans-serif" }}>
        <div style={{ maxWidth:640, margin:"0 auto" }}>

          {/* Header */}
          <div style={{ marginBottom:32 }}>
            <div style={{ display:"inline-block", background:"#111", color:"#f9f7f4",
              borderRadius:8, padding:"3px 10px", fontSize:11,
              fontFamily:"'DM Mono',monospace", letterSpacing:"0.1em",
              textTransform:"uppercase", marginBottom:14 }}>
              ML · TF-IDF · Multi-Model
            </div>
            <h1 style={{ fontFamily:"'DM Serif Display',serif", fontSize:38, fontWeight:400,
              margin:"0 0 8px", color:"#111", lineHeight:1.15 }}>
              Fake News<br /><em>Detector</em>
            </h1>
            <p style={{ fontSize:15, color:"#6b7280", margin:0, lineHeight:1.6 }}>
              Classify any news article by URL or by pasting the text directly.
            </p>
          </div>

          {/* Input card */}
          <div style={{ background:"#fff", border:"1.5px solid #e5e7eb", borderRadius:16,
            padding:20, marginBottom:20 }}>

            {/* Mode tabs */}
            <div style={{ display:"flex", gap:4, background:"#f3f4f6", borderRadius:10,
              padding:4, marginBottom:18 }}>
              <button className={`mode-tab${mode==="url"?" active":""}`}
                onClick={() => { setMode("url"); reset(); }}>
                URL
              </button>
              <button className={`mode-tab${mode==="text"?" active":""}`}
                onClick={() => { setMode("text"); reset(); }}>
                Paste text
              </button>
            </div>

            {/* URL mode */}
            {mode === "url" && (
              <>
                <label style={{ fontSize:12, fontWeight:500, color:"#374151",
                  display:"block", marginBottom:8, letterSpacing:"0.04em" }}>
                  ARTICLE URL
                </label>
                <div style={{ display:"flex", gap:10, marginBottom:12 }}>
                  <input ref={inputRef} className="url-input" type="url"
                    placeholder="https://example.com/news/article"
                    value={url} onChange={e => setUrl(e.target.value)}
                    onKeyDown={e => e.key==="Enter" && canSubmit && analyze()}
                    disabled={status==="loading"}/>
                  <button className="analyze-btn" onClick={analyze} disabled={!canSubmit}>
                    {status==="loading" ? <Spinner/> : (
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8"
                          strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                    {status==="loading" ? "Analyzing…" : "Analyze"}
                  </button>
                </div>
                <div style={{ display:"flex", flexWrap:"wrap", gap:6, alignItems:"center" }}>
                  <span style={{ fontSize:11, color:"#9ca3af", letterSpacing:"0.06em",
                    textTransform:"uppercase" }}>Try:</span>
                  {SAMPLE_URLS.map(s => (
                    <button key={s} className="sample-btn" onClick={() => setUrl(s)}>
                      {s.replace(/^https?:\/\/(www\.)?/,'')}
                    </button>
                  ))}
                </div>
              </>
            )}

            {/* Text mode */}
            {mode === "text" && (
              <>
                <label style={{ fontSize:12, fontWeight:500, color:"#374151",
                  display:"block", marginBottom:8, letterSpacing:"0.04em" }}>
                  ARTICLE TITLE <span style={{ color:"#9ca3af", fontWeight:400 }}>(optional)</span>
                </label>
                <input className="title-input" type="text"
                  placeholder="Enter the article headline…"
                  value={articleTitle} onChange={e => setArticleTitle(e.target.value)}
                  style={{ marginBottom:12 }}/>

                <label style={{ fontSize:12, fontWeight:500, color:"#374151",
                  display:"block", marginBottom:8, letterSpacing:"0.04em" }}>
                  ARTICLE TEXT
                </label>
                <textarea ref={inputRef} className="text-input"
                  placeholder="Paste the full article body here. The more text, the more accurate the prediction…"
                  value={articleText} onChange={e => setArticleText(e.target.value)}
                  disabled={status==="loading"}/>
                <div className="word-count">
                  {articleText.trim() ? `${articleText.trim().split(/\s+/).length} words` : "min. 10 words"}
                </div>
                <div style={{ display:"flex", justifyContent:"flex-end", marginTop:10 }}>
                  <button className="analyze-btn" onClick={analyze} disabled={!canSubmit}>
                    {status==="loading" ? <Spinner/> : (
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8"
                          strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                    {status==="loading" ? "Analyzing…" : "Analyze"}
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Model selector */}
          <div style={{ position:"relative", marginBottom:20 }} ref={menuRef}>
            <div style={{ display:"flex", alignItems:"center", gap:10 }}>
              <span style={{ fontSize:12, color:"#6b7280", fontWeight:500,
                letterSpacing:"0.04em", textTransform:"uppercase", whiteSpace:"nowrap" }}>
                Model
              </span>
              <button className="model-btn" onClick={() => setShowModelMenu(v => !v)}
                style={{ flex:1 }}>
                <span style={{ flex:1, textAlign:"left" }}>{selectedModel}</span>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"
                  style={{ transform: showModelMenu ? "rotate(180deg)" : "none", transition:"transform .2s" }}>
                  <path d="M3 5l4 4 4-4" stroke="#6b7280" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
            {showModelMenu && (
              <div className="model-menu">
                {(availableModels.length ? availableModels : Object.keys(MODEL_DESCRIPTIONS)).map(name => (
                  <button key={name} className={`model-option${name===selectedModel?" selected":""}`}
                    onClick={() => { setSelectedModel(name); setShowModelMenu(false); }}>
                    {name}
                    <span className="model-option-desc">{MODEL_DESCRIPTIONS[name]}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Loading */}
          {status === "loading" && (
            <div style={{ display:"flex", alignItems:"center", gap:12, color:"#6b7280",
              fontSize:14, padding:"4px 0", animation:"fadeUp .3s ease" }}>
              <Spinner/>
              <span>
                {mode==="url"
                  ? "Fetching page → extracting text → running classifier…"
                  : `Running ${selectedModel} classifier…`}
              </span>
            </div>
          )}

          {/* Error */}
          {status === "error" && (
            <div style={{ background:"#fef2f2", border:"1.5px solid #fecaca", borderRadius:12,
              padding:"14px 18px", animation:"fadeUp .3s ease",
              display:"flex", gap:12, alignItems:"flex-start" }}>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none"
                style={{ flexShrink:0, marginTop:1 }} aria-hidden="true">
                <circle cx="9" cy="9" r="8" stroke="#ef4444" strokeWidth="1.5"/>
                <path d="M9 5v4M9 12.5v.5" stroke="#ef4444" strokeWidth="1.8" strokeLinecap="round"/>
              </svg>
              <div>
                <p style={{ margin:"0 0 4px", fontWeight:500, fontSize:14, color:"#dc2626" }}>Analysis failed</p>
                <p style={{ margin:0, fontSize:13, color:"#b91c1c" }}>{errorMsg}</p>
              </div>
              <button className="reset-btn" style={{ marginLeft:"auto", flexShrink:0 }} onClick={reset}>
                Try again
              </button>
            </div>
          )}

          {/* Result */}
          {status === "result" && result && (
            <div className="result-card">

              <div style={{ display:"flex", alignItems:"flex-start",
                justifyContent:"space-between", marginBottom:20, gap:12 }}>
                <div>
                  <div style={{ display:"flex", alignItems:"center", gap:8, flexWrap:"wrap" }}>
                    <Pill label={`Likely ${verdict.word}`} color={verdict.color} bg={verdict.bg}/>
                    <span style={{ fontSize:11, color:"#9ca3af", fontFamily:"'DM Mono',monospace" }}>
                      via {result.model_used}
                    </span>
                  </div>
                  <p style={{ margin:"10px 0 0", fontFamily:"'DM Serif Display',serif",
                    fontSize:22, color:"#111", lineHeight:1.3 }}>
                    {result.title || result.domain}
                  </p>
                </div>
                <button className="reset-btn" onClick={reset} style={{ flexShrink:0 }}>New URL</button>
              </div>

              <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:24 }}>
                <span className="meta-chip">{result.domain}</span>
                <span className="meta-chip">{result.word_count.toLocaleString()} words</span>
                <span className="meta-chip">{mode === "text" ? "pasted text" : "scraped"}</span>
              </div>

              <div style={{ display:"flex", justifyContent:"center", gap:32,
                background:"#f9f7f4", borderRadius:12, padding:"20px 24px", marginBottom:20 }}>
                <ConfidenceRing value={result.fake_prob} label="Fake" color="#ef4444"/>
                <div style={{ width:1, background:"#e5e7eb" }}/>
                <ConfidenceRing value={result.real_prob} label="Real" color="#10b981"/>
                <div style={{ width:1, background:"#e5e7eb" }}/>
                <ConfidenceRing value={result.confidence} label="Confidence" color={verdict.ring}/>
              </div>

              <div style={{ marginBottom:20 }}>
                <div style={{ display:"flex", justifyContent:"space-between",
                  marginBottom:6, fontSize:12, color:"#6b7280" }}>
                  <span>Classifier confidence</span>
                  <span style={{ fontFamily:"'DM Mono',monospace", fontWeight:500,
                    color:"#111" }}>{result.confidence}%</span>
                </div>
                <div style={{ height:7, background:"#e5e7eb", borderRadius:100, overflow:"hidden" }}>
                  <div style={{ height:"100%", borderRadius:100, width:`${result.confidence}%`,
                    background: result.confidence > 80
                      ? (isReal ? "#10b981" : "#ef4444") : "#f59e0b",
                    transition:"width 0.9s cubic-bezier(.4,0,.2,1)" }}/>
                </div>
                <p style={{ fontSize:11, color:"#9ca3af", margin:"5px 0 0" }}>
                  {result.confidence >= 85
                    ? "High confidence — the model is very certain about this classification."
                    : result.confidence >= 65
                      ? "Moderate confidence — review the article preview for context."
                      : "Low confidence — treat this result with caution."}
                </p>
              </div>

              <div>
                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                  <span style={{ fontSize:12, fontWeight:500, color:"#374151",
                    letterSpacing:"0.04em", textTransform:"uppercase" }}>
                    Article preview
                  </span>
                  <button className="toggle-preview" onClick={() => setShowPreview(v => !v)}>
                    {showPreview ? "Hide" : "Show"} preview
                  </button>
                </div>
                {showPreview && (
                  <div className="preview-text" style={{ animation:"fadeUp .25s ease" }}>
                    {result.preview}
                  </div>
                )}
              </div>
            </div>
          )}

          <p style={{ fontSize:12, color:"#9ca3af", marginTop:24, textAlign:"center", lineHeight:1.6 }}>
            Backend: Flask · Models: {Object.keys(MODEL_DESCRIPTIONS).join(", ")}<br/>
            Run <code style={{ fontFamily:"'DM Mono',monospace", background:"#e5e7eb",
              padding:"1px 5px", borderRadius:4 }}>python -m src.app</code> from <code
              style={{ fontFamily:"'DM Mono',monospace", background:"#e5e7eb",
              padding:"1px 5px", borderRadius:4 }}>backend/</code>
          </p>

        </div>
      </div>
    </>
  );
}