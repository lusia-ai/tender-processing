"use client";

import { DragEvent, FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Source = { source: string; page?: number; score: number; text: string };
type ChatResponse = { thread_id: string; answer: string; sources: Source[] };
type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  attachment?: { name: string };
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Home() {
  const maxK = 20;
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState(
    "Summarize the scope, timeline, and key requirements for tender ted_812-2018_EN.pdf."
  );
  const [sourceFilter, setSourceFilter] = useState("");
  const [topK, setTopK] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const onFileSelected = (file: File | null) => {
    if (!file) {
      setAttachedFile(null);
      return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    setError(null);
    setAttachedFile(file);
  };

  const onDropFile = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0] || null;
    onFileSelected(file);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setError(null);
    try {
      const userMessage: Message = {
        role: "user",
        content: input,
        attachment: attachedFile ? { name: attachedFile.name } : undefined,
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");

      const isFileUpload = Boolean(attachedFile);
      const endpoint = isFileUpload ? `${API_BASE}/chat-file` : `${API_BASE}/chat`;
      const res = await fetch(endpoint, {
        method: "POST",
        headers: isFileUpload ? undefined : { "Content-Type": "application/json" },
        body: isFileUpload
          ? (() => {
              const form = new FormData();
              form.append("message", userMessage.content);
              if (threadId) form.append("thread_id", threadId);
              if (sourceFilter) form.append("source_filter", sourceFilter);
              form.append("top_k", String(topK));
              form.append("file", attachedFile as File);
              return form;
            })()
          : JSON.stringify({
              message: userMessage.content,
              thread_id: threadId,
              source_filter: sourceFilter || null,
              top_k: topK,
            }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Request failed");
      }
      const json = (await res.json()) as ChatResponse;
      setThreadId(json.thread_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: json.answer, sources: json.sources },
      ]);
      setAttachedFile(null);
    } catch (err: any) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const rangeFill = Math.round((topK / maxK) * 100);

  return (
    <main className="page">
      <div className="shell">
        <header className="hero">
          <div>
            <p className="eyebrow">Tender Intelligence</p>
            <h1>Procurement Briefing Console</h1>
            <p className="sub">
              Query internal tender records with traceable sources. Designed for bid and compliance teams.
            </p>
          </div>
          <div className="meta">
            <div>
              <span className="meta-label">Environment</span>
              <span className="meta-value">Production Sandbox</span>
            </div>
            <div>
              <span className="meta-label">Index</span>
              <span className="meta-value">port / tenders</span>
            </div>
            <div>
              <span className="meta-label">Database</span>
              <span className="meta-value">db / tender_company</span>
            </div>
          </div>
        </header>

        <section className="workspace">
          <div className="chat-panel">
            <div className="chat-log">
              {messages.length === 0 && (
                <div className="empty-state">
                  <p>Start a conversation to generate a tender briefing.</p>
                  <p className="note">Context and sources are retained within this session.</p>
                </div>
              )}
              {messages.map((msg, idx) => (
                <div key={`${msg.role}-${idx}`} className={`message ${msg.role}`}>
                  <div className="message-label">{msg.role === "user" ? "You" : "Assistant"}</div>
                  <div className="message-body prose">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                  {msg.attachment && (
                    <div className="message-attachment">
                      <span className="attachment-icon">📎</span>
                      <span className="attachment-name">{msg.attachment.name}</span>
                    </div>
                  )}
                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                    <details className="message-sources">
                      <summary className="sources-title">
                        <span>Sources</span>
                        <span className="sources-meta">{msg.sources.length} items</span>
                      </summary>
                      <ul>
                        {msg.sources.map((s, sourceIdx) => (
                          <li key={`${s.source}-${s.page}-${sourceIdx}`}>
                            <span className="pill">
                              {s.source}
                              {s.page ? `#p${s.page}` : ""}
                            </span>
                            <span className="score">{s.score.toFixed(3)}</span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              ))}
            </div>

            <form className="composer" onSubmit={onSubmit}>
              <div
                className={`composer-shell ${dragActive ? "is-dragging" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={onDropFile}
              >
                {attachedFile && (
                  <div className="composer-toolbar">
                    <div className="file-chip">
                      <span>{attachedFile.name}</span>
                      <button
                        type="button"
                        className="file-remove"
                        onClick={() => setAttachedFile(null)}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                )}
                <div className="composer-body">
                  <label className="clip-button icon-button secondary" title="Attach tender PDF">
                    <span className="clip-icon">📎</span>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(event) => onFileSelected(event.target.files?.[0] || null)}
                    />
                  </label>
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    rows={3}
                    placeholder="Ask about scope, timelines, award criteria, or qualifications..."
                    required
                    onDrop={onDropFile}
                    onDragOver={(event) => event.preventDefault()}
                  />
                  <div className="composer-actions">
                    <button
                      type="button"
                      className="clip-button icon-button secondary"
                      aria-label="Open session settings"
                      onClick={() => setSettingsOpen(true)}
                    >
                      <span className="clip-icon" aria-hidden="true">
                        ⚙︎
                      </span>
                    </button>
                    <button
                      type="submit"
                      className="icon-button primary"
                      aria-label="Send message"
                      disabled={loading}
                    >
                      <span aria-hidden="true">{loading ? "…" : "➤"}</span>
                    </button>
                  </div>
                </div>
              </div>
            </form>
            {error && <div className="error">⚠️ {error}</div>}
          </div>
        </section>

        {settingsOpen && (
          <div className="settings-overlay" onClick={() => setSettingsOpen(false)}>
            <aside className="status settings-panel" onClick={(event) => event.stopPropagation()}>
              <div className="card-head">
                <h3>Session controls</h3>
                <div className="settings-actions">
                  <span className="badge">Live</span>
                  <button type="button" className="icon-button text" onClick={() => setSettingsOpen(false)}>
                    Close
                  </button>
                </div>
              </div>
              <p className="note">Refine retrieval depth or narrow to a single tender file.</p>
              <div className="divider" />
              <div className="control-group">
                <label className="label">
                  Source file
                  <input
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    placeholder="Optional: ted_812-2018_EN.pdf"
                  />
                </label>
              </div>
              <div className="control-group">
                <div className="control-row">
                  <span className="label-title">Retrieval depth</span>
                  <span className="value-pill">{topK}</span>
                </div>
                <div className="range-wrap">
                  <input
                    className="range"
                    type="range"
                    min={1}
                    max={maxK}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    style={{
                      background: `linear-gradient(90deg, var(--accent) ${rangeFill}%, rgba(255, 255, 255, 0.12) ${rangeFill}%)`,
                    }}
                  />
                  <div className="range-scale">
                    <span>1</span>
                    <span>{maxK}</span>
                  </div>
                </div>
              </div>
            </aside>
          </div>
        )}
      </div>
    </main>
  );
}
