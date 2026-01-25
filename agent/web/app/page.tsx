"use client";

import { DragEvent, FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Source = { source: string; page?: number; score: number; text: string };
type ToolOutput = { kind: string; format?: string; note?: string; content?: string; confidence?: number; data?: any };
type ChatResponse = {
  thread_id: string;
  answer: string;
  sources: Source[];
  tool_outputs?: ToolOutput[];
};
type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  toolOutputs?: ToolOutput[];
  attachment?: { name: string };
};

type Delivery = {
  tender_id?: string;
  title?: string;
  customer?: string;
  delivered_at?: string;
  value?: number;
  currency?: string;
  scope?: string;
};

type TenderBreakdownSection = { title?: string; bullets?: string[]; sources?: string[] };
const DEBUG_UI = process.env.NEXT_PUBLIC_DEBUG_UI === "true";

if (DEBUG_UI) {
  console.info("DEBUG_UI enabled");
}

type TenderBreakdownData = {
  title?: string;
  sections?: TenderBreakdownSection[];
  company_deliveries?: Delivery[];
  company_delivery_summary?: string;
};

type SimilarSnippet = { page?: number; score?: number; snippet?: string };
type SimilarMatch = {
  tender_id?: string;
  title?: string;
  source?: string;
  top_score?: number;
  snippets?: SimilarSnippet[];
};

type SimilarData = {
  matches?: SimilarMatch[];
  queries?: string[];
  company_deliveries?: Delivery[];
  company_delivery_summary?: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type ReadinessLevel = "high" | "medium" | "low" | "uncertain" | "unknown";

type ReadinessMeta = { label: string; percent: number; tone: "positive" | "neutral" | "negative" };

const readinessMeta: Record<ReadinessLevel, ReadinessMeta> = {
  high: { label: "High", percent: 82, tone: "positive" },
  medium: { label: "Medium", percent: 62, tone: "neutral" },
  low: { label: "Low", percent: 35, tone: "negative" },
  uncertain: { label: "Uncertain", percent: 50, tone: "neutral" },
  unknown: { label: "Unknown", percent: 50, tone: "neutral" },

};

const readinessLabelsRu: Record<ReadinessLevel, string> = {
  high: "Высокая",
  medium: "Средняя",
  low: "Низкая",
  uncertain: "Неопределенная",
  unknown: "Неизвестная",
};


const headingMap = {
  summary: ["short summary", "summary", "краткое резюме", "краткий итог", "краткое саммари", "резюме"],
  reasons: ["key reasons", "reasons", "ключевые причины", "причины"],
  gaps: ["open gaps", "gaps", "unknowns", "risks", "пробелы", "риски"],
  next: ["next steps", "следующие шаги", "дальше"],
};

const verdictMatchers = [
  { level: "high" as const, tokens: ["high", "высок"] },
  { level: "medium" as const, tokens: ["medium", "средн"] },
  { level: "low" as const, tokens: ["low", "низк"] },
  { level: "uncertain" as const, tokens: ["uncertain", "неопредел", "условн", "сомн"] },
];

function detectReadinessLevel(text: string): ReadinessLevel {
  const lower = text.toLowerCase();
  for (const entry of verdictMatchers) {
    if (entry.tokens.some((token) => lower.includes(token))) {
      return entry.level;
    }
  }
  return "unknown";
}

function extractReadinessConfidence(text: string): number | null {
  const match = text.match(/(?:confidence|уверенность)\s*[:\-–—]?\s*(\d{1,3})\s*%/i);
  if (!match) return null;
  const value = Number.parseInt(match[1], 10);
  if (Number.isNaN(value)) return null;
  return Math.min(100, Math.max(0, value));
}

function clampReadinessConfidence(level: ReadinessLevel, value: number): number {
  const ranges: Record<ReadinessLevel, [number, number]> = {
    high: [80, 98],
    medium: [55, 79],
    low: [30, 54],
    uncertain: [40, 60],
    unknown: [40, 60],
  };
  const [min, max] = ranges[level] || [30, 95];
  return Math.max(min, Math.min(max, value));
}



function extractReadinessSections(text: string) {
  const lines = text.split("\n");
  const sections = { verdictLine: "", summary: [] as string[], reasons: [] as string[], gaps: [] as string[], next: [] as string[] };

  const verdictMatch = text.match(/\bVerdict\b[^\n]*/i) || text.match(/\bВердикт\b[^\n]*/i);
  if (verdictMatch) {
    sections.verdictLine = verdictMatch[0].replace(/\*/g, "").trim();
  }

  let current: keyof typeof headingMap | "" = "";
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    const clean = line.replace(/\*/g, "").toLowerCase();
    const matched = (Object.keys(headingMap) as Array<keyof typeof headingMap>).find((key) =>
      headingMap[key].some((label) => clean.startsWith(label))
    );
    if (matched) {
      current = matched;
      continue;
    }
    if (!current) continue;
    if (line.startsWith("-") || line.startsWith("•")) {
      const item = line.replace(/^[-•]\s*/, "").trim();
      if (item) sections[current].push(item);
    }
  }
  return sections;
}

function getLatestToolOutput(outputs: ToolOutput[] | undefined, kind: string): ToolOutput | undefined {
  if (!outputs || outputs.length === 0) return undefined;
  for (let i = outputs.length - 1; i >= 0; i -= 1) {
    if (outputs[i].kind === kind) return outputs[i];
  }
  return undefined;
}

function parseBreakdownFromText(text: string): TenderBreakdownData | null {
  if (!text) return null;
  const raw = text.replace(/\r/g, "");
  const lines = raw.split("\n").map((line) => line.trim()).filter(Boolean);
  if (lines.length === 0) return null;
  const firstLine = lines[0].toLowerCase();
  const hasHeader = firstLine.includes("разбор тендера") || firstLine.includes("tender breakdown");
  const normalizeLine = (line: string) => line.replace(/^[-•–—]\s*/, "");
  const hasNumbered = lines.some((line) => /^\d+[).]/.test(normalizeLine(line)))
    || /^\s*\d+[).]\s/.test(raw);
  if (!hasHeader && !hasNumbered) return null;

  let title = lines[0];
  if (hasHeader) {
    title = lines[0]
      .replace(/^(Разбор тендера|Tender breakdown)\s*[:\-–—]?\s*/i, "")
      .trim();
  }

  const sections: TenderBreakdownSection[] = [];
  let current: TenderBreakdownSection | null = null;

  for (const line of lines) {
    const normalized = normalizeLine(line);
    const match = normalized.match(/^(\d+)[).]\s*(.*)/);
    if (match) {
      if (current) sections.push(current);
      const sectionTitle = match[2] ? match[2].trim() : `Section ${match[1]}`;
      current = { title: sectionTitle, bullets: [] };
      continue;
    }
    if (!current) continue;
    current.bullets = current.bullets || [];
    current.bullets.push(normalized);
  }
  if (current) sections.push(current);

  if (sections.length <= 1 && raw.length > 0) {
    const chunks = raw
      .split(/(?=\d+[).]\s)/g)
      .map((chunk) => chunk.trim())
      .filter(Boolean);
    if (chunks.length > 1) {
      const rebuilt: TenderBreakdownSection[] = [];
      for (const chunk of chunks) {
        const normalized = normalizeLine(chunk);
        const match = normalized.match(/^(\d+)[).]\s*(.*)/);
        const body = match ? match[2] : normalized;
        const parts = body.split("\n").map((item) => item.trim()).filter(Boolean);
        const sectionTitle = parts[0] || "Section";
        const rest = parts.slice(1);
        rebuilt.push({ title: sectionTitle, bullets: rest });
      }
      if (rebuilt.length > sections.length) {
        return { title: title || undefined, sections: rebuilt };
      }
    }
  }

  if (sections.length === 0 && lines.length > 1) {
    sections.push({ title: "Summary", bullets: lines.slice(1).map(normalizeLine) });
  }

  return { title: title || undefined, sections };
}

const renderInline = (value: string) => (
  <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: "span" }}>
    {value}
  </ReactMarkdown>
);

const stripMarkdown = (value: string) =>
  value.replace(/\*\*/g, "").replace(/^#+\s*/, "").replace(/^\d+[).]\s*/, "").trim();


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
        {
          role: "assistant",
          content: json.answer,
          sources: json.sources,
          toolOutputs: json.tool_outputs || [],
        },
      ]);
      setAttachedFile(null);
    } catch (err: any) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const rangeFill = Math.round((topK / maxK) * 100);

  const renderReadinessCard = (msg: Message, output: ToolOutput) => {
    const text = output.content || msg.content;
    const sections = extractReadinessSections(text);
    const level = detectReadinessLevel(sections.verdictLine || text);
    const meta = readinessMeta[level];
    const confidenceOverride = output.confidence ?? extractReadinessConfidence(text);
    const rawPercent = confidenceOverride ?? meta.percent;
    const percent = clampReadinessConfidence(level, rawPercent);
    const toneClass = `tone-${meta.tone}`;
    const isRussian = /[А-Яа-яЁё]/.test(text);
    const levelLabel = isRussian ? readinessLabelsRu[level] : meta.label;
    const labels = {
      kicker: isRussian ? "Готовность компании" : "Company readiness",
      confidence: isRussian ? "уверенность" : "confidence",
      reasons: isRussian ? "Ключевые причины" : "Key reasons",
      gaps: isRussian ? "Пробелы" : "Open gaps",
      next: isRussian ? "Следующие шаги" : "Next steps",
      memo: isRussian ? "Подробный анализ" : "Detailed analysis",
      sources: isRussian ? "Источники" : "Sources",
    };

    return (
      <div className={`readiness-card ${toneClass}`}>
        <div className="readiness-header">
          <div>
            <p className="readiness-kicker">{labels.kicker}</p>
            <p className="readiness-title">{levelLabel} {labels.confidence}</p>
            {sections.verdictLine && <p className="readiness-verdict">{sections.verdictLine}</p>}
          </div>
          <div className="readiness-score">
            <span className="readiness-percent">{percent}%</span>
            <span className="readiness-label">{levelLabel}</span>
          </div>
        </div>
        <div className="readiness-meter">
          <span style={{ width: `${percent}%` }} />
        </div>
        <div className="readiness-sections">
          {sections.summary.length > 0 && (
            <div className="readiness-summary">
              <p className="readiness-summary-label">{isRussian ? "Коротко" : "Short summary"}</p>
              <ul>
                {sections.summary.map((item, idx) => (
                  <li key={`summary-${idx}`}>{renderInline(item)}</li>
                ))}
              </ul>
            </div>
          )}
          {sections.reasons.length > 0 && (
            <details className="readiness-section">
              <summary>{labels.reasons}</summary>
              <ul>
                {sections.reasons.map((item, idx) => (
                  <li key={`reason-${idx}`}>{renderInline(item)}</li>
                ))}
              </ul>
            </details>
          )}
          {sections.gaps.length > 0 && (
            <details className="readiness-section">
              <summary>{labels.gaps}</summary>
              <ul>
                {sections.gaps.map((item, idx) => (
                  <li key={`gap-${idx}`}>{renderInline(item)}</li>
                ))}
              </ul>
            </details>
          )}
          {sections.next.length > 0 && (
            <details className="readiness-section">
              <summary>{labels.next}</summary>
              <ul>
                {sections.next.map((item, idx) => (
                  <li key={`next-${idx}`}>{renderInline(item)}</li>
                ))}
              </ul>
            </details>
          )}
          <details className="readiness-section">
            <summary>{labels.memo}</summary>
            <div className="readiness-body prose">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
            </div>
          </details>
          {msg.sources && msg.sources.length > 0 && (
            <details className="readiness-section">
              <summary>{labels.sources} ({msg.sources.length})</summary>
              <ul className="readiness-sources">
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
      </div>
    );
  };


  const renderTenderBreakdownCard = (msg: Message, output: ToolOutput) => {
    const data = output.data as TenderBreakdownData | undefined;
    const text = output.content || msg.content;
    const isRussian = /[А-Яа-яЁё]/.test(text + JSON.stringify(data || {}));
    const sections = (data?.sections || []).map((section) => ({
      ...section,
      title: stripMarkdown(section.title || ""),
    }));
    const title = stripMarkdown(
      data?.title || (isRussian ? "Разбор тендера" : "Tender breakdown")
    );
    const labels = {
      kicker: isRussian ? "Разбор тендера" : "Tender breakdown",
      memo: isRussian ? "Полный отчет" : "Full memo",
      sources: isRussian ? "Источники" : "Sources",
      deliveries: isRussian ? "Релевантные поставки" : "Relevant deliveries",
      empty: isRussian ? "Нет структурированных секций" : "No structured sections",
      noDetails: isRussian ? "Нет данных" : "No details found",
    };

    const sectionLabel = isRussian ? "секций" : "sections";

    if (DEBUG_UI) {
      console.info("tender_breakdown", {
        baseSections: sections.length,
        sectionTitles: sections.map((section) => section.title),
        toolNote: output.note,
        hasData: Boolean(data),
      });
    }

    const memoContent = output.content || msg.content;

    return (
      <div className="info-card tender-card">
        <div className="card-header">
          <div>
            <p className="card-kicker">{labels.kicker}</p>
            <p className="card-title">{title}</p>
            <p className="card-subtitle">{sections.length} {sectionLabel}</p>
          </div>
        </div>
        <div className="card-sections">
          {sections.length === 0 && <p className="card-empty">{labels.empty}</p>}
          {sections.map((section, idx) => {
            const bulletCount = section.bullets?.length || 0;
            return (
              <details className="card-section" key={`section-${idx}`}>
                <summary>
                  <span>{section.title || (isRussian ? "Раздел" : "Section")}</span>
                </summary>
                {bulletCount > 0 ? (
                  <ul>
                    {section.bullets?.map((item, itemIdx) => (
                      <li key={`section-${idx}-${itemIdx}`}>{renderInline(item)}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="section-body">{labels.noDetails}</div>
                )}
                {section.sources && section.sources.length > 0 && (
                  <div className="section-sources">
                    {section.sources.map((source, sourceIdx) => (
                      <span className="pill" key={`section-${idx}-src-${sourceIdx}`}>
                        {source}
                      </span>
                    ))}
                  </div>
                )}
              </details>
            );
          })}
          {data?.company_deliveries && data.company_deliveries.length > 0 && (
            <details className="card-section">
              <summary>
                <span>{labels.deliveries}</span>
                <span className="section-meta">{data.company_deliveries.length}</span>
              </summary>
              <ul>
                {data.company_deliveries.map((delivery, idx) => (
                  <li key={`delivery-${idx}`}>
                    {delivery.delivered_at ? `${delivery.delivered_at} · ` : ""}
                    {delivery.title || ""}
                    {delivery.value ? ` · ${delivery.value} ${delivery.currency || ""}` : ""}
                    {delivery.customer ? ` · ${delivery.customer}` : ""}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {data?.company_delivery_summary && (!data.company_deliveries || data.company_deliveries.length === 0) && (
            <details className="card-section">
              <summary>{labels.deliveries}</summary>
              <div className="section-body">{renderInline(data.company_delivery_summary)}</div>
            </details>
          )}
          <details className="card-section">
            <summary>{labels.memo}</summary>
            <div className="card-body prose">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{memoContent}</ReactMarkdown>
            </div>
          </details>
          {msg.sources && msg.sources.length > 0 && (
            <details className="card-section">
              <summary>
                <span>{labels.sources}</span>
                <span className="section-meta">{msg.sources.length}</span>
              </summary>
              <ul className="card-sources">
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
      </div>
    );
  };

  const renderSimilarCard = (msg: Message, output: ToolOutput) => {
    const data = output.data as SimilarData | undefined;
    const matches = data?.matches || [];
    const text = output.content || msg.content;
    const isRussian = /[А-Яа-яЁё]/.test(text + JSON.stringify(data || {}));
    const labels = {
      kicker: isRussian ? "Похожие тендеры" : "Similar tenders",
      memo: isRussian ? "Полный отчет" : "Full memo",
      sources: isRussian ? "Источники" : "Sources",
      deliveries: isRussian ? "Релевантные поставки" : "Relevant deliveries",
      empty: isRussian ? "Нет совпадений" : "No matches found",
      queries: isRussian ? "Запросы" : "Queries",
    };

    const matchLabel = isRussian ? "совпадений" : "matches";

    return (
      <div className="info-card similar-card">
        <div className="card-header">
          <div>
            <p className="card-kicker">{labels.kicker}</p>
            <p className="card-title">{matches.length} {matchLabel}</p>
            {data?.queries && data.queries.length > 0 && (
              <p className="card-subtitle">{labels.queries}: {data.queries.slice(0, 3).join(", ")}</p>
            )}
          </div>
        </div>
        <div className="similar-list">
          {matches.length === 0 && <p className="card-empty">{labels.empty}</p>}
          {matches.map((match, idx) => (
            <div className="similar-item" key={`match-${idx}`}>
              <div className="similar-head">
                <div>
                  <p className="similar-title">{match.title || match.tender_id || match.source || "Tender"}</p>
                  <p className="similar-subtitle">
                    {match.source}
                  </p>
                </div>
                {typeof match.top_score === "number" && (
                  <span className="similar-score">{match.top_score.toFixed(2)}</span>
                )}
              </div>
              {match.snippets && match.snippets.length > 0 ? (
                <ul className="similar-snippets">
                  {match.snippets.map((snippet, snippetIdx) => (
                    <li className="similar-snippet" key={`snippet-${idx}-${snippetIdx}`}>
                      <span>{renderInline(snippet.snippet || "")}</span>
                      <span className="match-meta">
                        {match.source}
                        {snippet.page ? `#p${snippet.page}` : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="card-empty">{labels.empty}</p>
              )}
            </div>
          ))}
        </div>
        {data?.company_deliveries && data.company_deliveries.length > 0 && (
          <div className="similar-deliveries">
            <p className="similar-label">{labels.deliveries}</p>
            <ul className="similar-snippets">
              {data.company_deliveries.map((delivery, idx) => (
                <li className="similar-snippet" key={`delivery-${idx}`}>
                  <span>
                    {delivery.delivered_at ? `${delivery.delivered_at} · ` : ""}
                    {delivery.title || ""}
                    {delivery.value ? ` · ${delivery.value} ${delivery.currency || ""}` : ""}
                    {delivery.customer ? ` · ${delivery.customer}` : ""}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {data?.company_delivery_summary && (!data.company_deliveries || data.company_deliveries.length === 0) && (
          <div className="similar-deliveries">
            <p className="similar-label">{labels.deliveries}</p>
            <div className="section-body">{renderInline(data.company_delivery_summary)}</div>
          </div>
        )}
        <div className="card-body prose">
          <p className="similar-label">{labels.memo}</p>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <div className="similar-sources">
            <p className="similar-label">{labels.sources}</p>
            <div className="section-sources">
              {msg.sources.map((s, sourceIdx) => (
                <span className="pill" key={`${s.source}-${s.page}-${sourceIdx}`}>
                  {s.source}
                  {s.page ? `#p${s.page}` : ""}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };



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
              {messages.map((msg, idx) => {
                const readinessOutput = msg.toolOutputs?.find(
                  (output) => output.kind === "company_readiness" && output.content
                );
                const breakdownOutput = getLatestToolOutput(msg.toolOutputs, "tender_breakdown");
                const similarOutput = msg.toolOutputs?.find((output) => output.kind === "similar");
                const showCards = msg.role === "assistant" && Boolean(
                  readinessOutput || breakdownOutput || similarOutput
                );
                if (DEBUG_UI && msg.toolOutputs && msg.toolOutputs.length > 0) {
                  console.info("tool_outputs", msg.toolOutputs.map((output) => ({ kind: output.kind, note: output.note, hasData: Boolean(output.data) })));
                }
                return (
                  <div
                    key={`${msg.role}-${idx}`}
                    className={`message ${msg.role}${showCards ? " is-card" : ""}`}
                  >
                    <div className="message-label">{msg.role === "user" ? "You" : "Assistant"}</div>
                    {showCards ? (
                      <div className="message-cards">
                        {readinessOutput && renderReadinessCard(msg, readinessOutput)}
                        {breakdownOutput && renderTenderBreakdownCard(msg, breakdownOutput)}
                        {similarOutput && renderSimilarCard(msg, similarOutput)}
                      </div>
                    ) : (
                      <div className="message-body prose">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                    {msg.attachment && (
                      <div className="message-attachment">
                        <span className="attachment-icon" aria-hidden="true">📎</span>
                        <span className="attachment-name">{msg.attachment.name}</span>
                      </div>
                    )}
                    {!showCards && msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
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
                );
              })}
              {loading && (
                <div className="message assistant is-typing">
                  <div className="message-label">Assistant</div>
                  <div className="message-body typing" aria-live="polite">
                    <span className="typing-text">Thinking</span>
                    <span className="typing-dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                  </div>
                </div>
              )}
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
                      <span className="file-icon" aria-hidden="true">📎</span>
                      <span className="file-name">{attachedFile.name}</span>
                      <button
                        type="button"
                        className="file-remove"
                        onClick={() => setAttachedFile(null)}
                        aria-label="Remove attached file"
                      >
                        ×
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
