import {
  ChangeEvent,
  Dispatch,
  FormEvent,
  SetStateAction,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import * as pdfjsLib from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import "./styles.css";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

type Paper = {
  id: number;
  title: string;
  file_name: string;
  page_count: number;
  status: string;
  error_message?: string | null;
  imported_at?: string;
  vectorized?: number;
  needs_ocr?: number;
  embedding_model?: string;
  embedding_dim?: number | null;
  progress_stage?: string;
  progress_current?: number;
  progress_total?: number;
};
type PaperStatus = Paper & { progress_ratio?: number };
type Page = { id: number; page_number: number; text: string };
type PaperDetails = { paper: Paper; pages: Page[] };
type SearchResult = {
  id: number;
  text: string;
  page_number: number;
  section?: string;
  heading?: string;
  score?: number | null;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  source?: "fts" | "vector" | "hybrid";
  paper_id?: number;
  paper_title?: string;
  sources?: string[];
};
type Source = {
  id: number;
  paper_id?: number | null;
  paper_title?: string;
  text: string;
  page_number: number;
  section?: string;
  heading?: string;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
};
type AgentCitation = {
  paper_id?: number | null;
  page: number;
  quote: string;
  claim?: string;
  verified?: boolean;
  level?: string;
};
type AgentStep = {
  index?: number;
  thought?: string;
  tool: string;
  args?: Record<string, unknown>;
  observation?: Record<string, unknown>;
  duration_ms?: number;
};
type ChatMessage = {
  id?: number;
  role: "user" | "assistant";
  content: string;
  citations?: AgentCitation[];
  sources?: Source[];
  trace?: AgentStep[];
  warnings?: string[];
  verified?: boolean;
  error?: string;
  pending?: boolean;
  created_at?: string;
};
type ChatSession = {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};
type UsageStats = {
  tokens: {
    calls: number;
    prompt: number;
    completion: number;
    total: number;
    estimated_calls: number;
    today_calls: number;
    today_total: number;
  };
  runs: {
    total: number;
    today_runs: number;
    verified: number;
    avg_steps: number;
    modes: { mode: string; count: number }[];
  };
  tools: { total: number; by_tool: { tool: string; count: number }[] };
  models: {
    provider: string;
    model: string;
    calls: number;
    total_tokens: number;
    estimated_calls: number;
  }[];
  sessions: { total: number; messages: number };
  papers: { total: number; needs_ocr: number; vectorized: number };
};
type LogEvent = {
  seq: number;
  time: string;
  level: string;
  logger: string;
  message: string;
  event?: string;
  [key: string]: unknown;
};
type AgentDone = {
  id?: number;
  content?: string;
  citations?: AgentCitation[];
  sources?: Source[];
  trace?: AgentStep[];
  verified?: boolean;
  citation_level?: string;
  warnings?: string[];
};
type AgentEvent =
  | { type: "start"; mode: string; paper_id?: number | null }
  | { type: "delta"; text: string }
  | ({ type: "step" } & AgentStep)
  | ({ type: "done" } & AgentDone)
  | { type: "error"; error: string };
type StructuredField = {
  value: string;
  label?: string;
  verified?: boolean;
  evidence: Array<{ paper_id: number; page_number: number; quote?: string; verified?: boolean }>;
};
type Structured = {
  fields: Record<string, StructuredField>;
  source?: string;
  cached?: boolean;
  warnings?: string[];
  llm_available?: boolean;
};
type ProviderInfo = {
  id: string;
  label: string;
  protocol: "openai" | "anthropic" | "gemini";
  base_url: string;
  default_model: string;
  requires_key: boolean;
  supports_model_list: boolean;
};
type ProtocolInfo = {
  id: "openai" | "anthropic" | "gemini";
  label: string;
};
type Settings = {
  provider: string;
  provider_label: string;
  protocol: string;
  default_protocol: string;
  base_url: string;
  default_base_url: string;
  model: string;
  default_model: string;
  embedding_provider: string;
  embedding_model: string;
  ocr_enabled: string;
  ocr_engine: string;
  ocr_lang: string;
  ocr_dpi: string;
  ocr_endpoint: string;
  ocr_type: string;
  ocr_access_key_id: string;
  ocr_key_configured: boolean;
  ocr_env_configured: boolean;
  ocr_available: string;
  rerank_enabled: string;
  rerank_engine: string;
  rerank_model: string;
  rerank_base_url: string;
  rerank_key_configured: boolean;
  relevance_min_coverage: string;
  nli_enabled: string;
  nli_engine: string;
  nli_model: string;
  context_token_budget: string;
  api_key_configured: boolean;
  embedding_key_configured: boolean;
  providers: ProviderInfo[];
  protocols: ProtocolInfo[];
  data_dir: string;
};
type Highlight = { page: number; bbox: [number, number, number, number] | null };
type JumpHandler = (page: number, paperId?: number, bbox?: [number, number, number, number]) => void;
type GeneralMode = "library" | "general";

const API = "http://127.0.0.1:8765/api";

async function readResponse<T>(response: Response): Promise<T> {
  const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
  if (!response.ok) throw new Error(payload?.detail || `请求失败（HTTP ${response.status}）`);
  return payload as T;
}

async function readError(response: Response): Promise<string> {
  const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
  return payload?.detail || `请求失败（HTTP ${response.status}）`;
}

function reasonOf(error: unknown): string {
  return error instanceof Error ? error.message : "未知错误";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function bboxOf(result: { x0?: number; y0?: number; x1?: number; y1?: number }): [number, number, number, number] | undefined {
  if (result.x0 == null || result.y0 == null || result.x1 == null || result.y1 == null) return undefined;
  return [result.x0, result.y0, result.x1, result.y1];
}

async function consumeAgentStream(response: Response, onEvent: (event: AgentEvent) => void): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("当前浏览器不支持流式读取");
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed) as AgentEvent);
      } catch {
        /* 忽略未完成的半行 */
      }
    }
  }
  if (buffer.trim()) {
    try {
      onEvent(JSON.parse(buffer.trim()) as AgentEvent);
    } catch {
      /* 最后一行可能被截断 */
    }
  }
}

function stageLabel(stage?: string): string {
  const labels: Record<string, string> = {
    queued: "排队",
    starting: "准备",
    parsing: "解析页面",
    embedding: "生成向量",
    ready: "完成",
    failed: "失败",
    index_failed: "索引失败",
  };
  return labels[stage || ""] || "处理中";
}

function toolLabel(tool: string): string {
  const labels: Record<string, string> = {
    list_papers: "列出论文",
    search_library: "检索证据",
    read_page: "阅读页面",
    read_section: "阅读章节",
    compare_papers: "对比论文",
    extract_paper: "结构化提取",
    calculate: "计算",
    clinical_calculator: "临床评分",
    lookup_ontology: "概念查询",
    appraise_evidence: "证据评价",
    evidence_table: "证据表格",
    remember_note: "保存笔记",
    recall_notes: "检索笔记",
    forget_note: "删除笔记",
    finish: "生成回答",
    invalid_json: "修正输出格式",
  };
  return labels[tool] || tool;
}

function argsSummary(args?: Record<string, unknown>): string {
  if (!args) return "";
  const preferred = args.query ?? args.q ?? args.heading ?? args.section ?? args.aspects ?? args.question;
  if (preferred != null) return String(preferred).slice(0, 80);
  const text = JSON.stringify(args);
  return text.length > 80 ? `${text.slice(0, 80)}…` : text;
}

function observationSummary(observation?: Record<string, unknown>): string {
  if (!observation) return "";
  if (observation.ok === false) return `错误：${String(observation.error ?? "未知错误")}`;
  if (observation.verified) return "引用核验通过";
  if (typeof observation.count === "number") return `${observation.count} 条结果`;
  if (Array.isArray(observation.papers)) return `${observation.papers.length} 篇论文`;
  return "完成";
}

function Markdown({ content, onJump }: { content: string; onJump?: JumpHandler }) {
  const linked = content.replace(
    /\[(?:Paper\s*(\d+)\s*[·\-–]\s*)?Page\s*(\d+)\]/gi,
    (_match, paper: string | undefined, page: string) =>
      `[PDF 第 ${page} 页](#${paper ? `paper-${paper}-` : ""}page-${page})`,
  );
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const match = href?.match(/^#(?:paper-(\d+)-)?page-(\d+)$/);
            if (match && onJump) {
              return (
                <button
                  type="button"
                  className="page-link"
                  onClick={() => onJump(Number(match[2]), match[1] ? Number(match[1]) : undefined)}
                >
                  {children}
                </button>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
          table: ({ children }) => <div className="table-wrap"><table>{children}</table></div>,
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}

function AgentTrace({ message }: { message: ChatMessage }) {
  const steps = message.trace || [];
  const warnings = message.warnings || [];
  if (!steps.length && !warnings.length && message.verified == null) return null;
  return (
    <details className="trace" open={Boolean(message.pending)}>
      <summary>
        <span>执行轨迹 · {steps.length} 步</span>
        {message.verified != null && (message.citations?.length ?? 0) > 0 && (
          <em className={message.verified ? "verified" : "unverified"}>
            {message.verified ? "引用已核验" : "引用未核验"}
          </em>
        )}
      </summary>
      {steps.length > 0 && (
        <ol>
          {steps.map((step, index) => (
            <li key={`${step.index ?? index}-${step.tool}`}>
              <b>{step.index ?? index + 1}. {toolLabel(step.tool)}</b>
              {step.thought && <p className="trace-thought">{step.thought}</p>}
              {step.args && Object.keys(step.args).length > 0 && (
                <p className="trace-detail">输入：{argsSummary(step.args)}</p>
              )}
              <p className="trace-detail">
                结果：{observationSummary(step.observation)}
                {step.duration_ms != null && ` · ${step.duration_ms} ms`}
              </p>
            </li>
          ))}
        </ol>
      )}
      {warnings.map((warning, index) => (
        <p className="trace-warning" key={`${warning}-${index}`}>⚠ {warning}</p>
      ))}
    </details>
  );
}

function CitationChips({ citations, onJump }: { citations: AgentCitation[]; onJump: JumpHandler }) {
  return (
    <div className="citation-list">
      {citations.map((citation, index) => (
        <button
          type="button"
          className="citation-chip"
          key={`${citation.paper_id ?? "p"}-${citation.page}-${index}`}
          title={citation.quote}
          onClick={() => onJump(citation.page, citation.paper_id ?? undefined)}
        >
          <b>{citation.verified === false ? "!" : "✓"}</b> PDF 第 {citation.page} 页
          {citation.claim ? ` · ${citation.claim}` : ""}
        </button>
      ))}
    </div>
  );
}

function MessageView({
  message,
  onJump,
  onRetry,
}: {
  message: ChatMessage;
  onJump: JumpHandler;
  onRetry: () => void;
}) {
  const citations = message.citations ?? [];
  // Only surface evidence that actually backs a citation; retrieved chunks
  // the model never cited are noise (especially for greetings).
  const citedSources = (message.sources ?? []).filter((source, index, all) => {
    const cited = citations.some((citation) =>
      citation.page === source.page_number
      && (citation.paper_id == null || source.paper_id == null
        || citation.paper_id === source.paper_id));
    if (!cited) return false;
    return all.findIndex((other) =>
      other.page_number === source.page_number && other.paper_id === source.paper_id) === index;
  });
  return (
    <div className={`message ${message.role}`}>
      <span>{message.role === "user" ? "你" : "觅心"}</span>
      {message.role === "assistant" && message.pending && !message.content ? (
        <p className="typing">正在检索证据并核验引用…</p>
      ) : (
        <Markdown content={message.content} onJump={onJump} />
      )}
      <AgentTrace message={message} />
      {message.citations && message.citations.length > 0 && (
        <CitationChips citations={message.citations} onJump={onJump} />
      )}
      {citedSources.length > 0 && (
        <div className="source-list">
          {citedSources.map((source) => (
            <button
              type="button"
              className="source-chip"
              key={`${source.id}-${source.page_number}`}
              onClick={() => onJump(source.page_number, source.paper_id ?? undefined, bboxOf(source))}
            >
              证据 · PDF 第 {source.page_number} 页
              {source.section && ` · ${source.section}`}：{source.text.slice(0, 90)}
            </button>
          ))}
        </div>
      )}
      {message.error && <button className="retry-button" onClick={onRetry}>↻ 重试</button>}
      {message.role === "assistant" && !message.pending && message.id != null && (
        <div className="export-links">
          <a href={`${API}/agent/runs/${message.id}/export?format=md`} download>导出 Markdown</a>
          <a href={`${API}/agent/runs/${message.id}/export?format=csv`} download>引用 CSV</a>
        </div>
      )}
    </div>
  );
}

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<PaperDetails | null>(null);
  const [tab, setTab] = useState<
    "chat" | "library" | "reader" | "stats" | "logs" | "settings"
  >("chat");
  const [readerTab, setReaderTab] = useState<"read" | "insights" | "chat">("read");
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState<"hybrid" | "keyword">("hybrid");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [highlight, setHighlight] = useState<Highlight | null>(null);
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [structured, setStructured] = useState<Structured | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    provider: "",
    provider_label: "",
    protocol: "",
    default_protocol: "",
    base_url: "",
    default_base_url: "",
    model: "",
    default_model: "",
    embedding_provider: "qwen",
    embedding_model: "text-embedding-v3",
    ocr_enabled: "true",
    ocr_engine: "auto",
    ocr_lang: "chi_sim+eng",
    ocr_dpi: "220",
    ocr_endpoint: "ocr-api.cn-hangzhou.aliyuncs.com",
    ocr_type: "Advanced",
    ocr_access_key_id: "",
    ocr_key_configured: false,
    ocr_env_configured: false,
    ocr_available: "",
    rerank_enabled: "true",
    rerank_engine: "lexical",
    rerank_model: "",
    rerank_base_url: "https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank",
    rerank_key_configured: false,
    relevance_min_coverage: "0",
    nli_enabled: "true",
    nli_engine: "none",
    nli_model: "",
    context_token_budget: "24000",
    api_key_configured: false,
    embedding_key_configured: false,
    providers: [],
    protocols: [],
    data_dir: "",
  });
  const [apiKey, setApiKey] = useState("");
  const [embeddingApiKey, setEmbeddingApiKey] = useState("");
  const [rerankApiKey, setRerankApiKey] = useState("");
  const [ocrAccessKeySecret, setOcrAccessKeySecret] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const [generalChat, setGeneralChat] = useState<ChatMessage[]>([]);
  const [generalQuestion, setGeneralQuestion] = useState("");
  const [generalMode, setGeneralMode] = useState<GeneralMode>("library");
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [libraryQuery, setLibraryQuery] = useState("");
  const [libraryResults, setLibraryResults] = useState<SearchResult[]>([]);
  const [librarySearching, setLibrarySearching] = useState(false);
  const agentAbort = useRef<AbortController | null>(null);

  const loadPapers = async () => {
    try {
      setPapers(await readResponse<Paper[]>(await fetch(`${API}/papers`)));
    } catch (error) {
      setNotice(`无法连接本地服务：${reasonOf(error)}`);
    }
  };
  const loadSettings = async () => {
    try {
      setSettings(await readResponse<Settings>(await fetch(`${API}/settings`)));
    } catch {
      /* older backend */
    }
  };
  const loadChatSessions = async (): Promise<ChatSession[]> => {
    try {
      const rows = await readResponse<ChatSession[]>(await fetch(`${API}/chat/sessions`));
      setChatSessions(rows);
      return rows;
    } catch {
      return [];
    }
  };
  const loadGeneralChat = async (sessionId: number) => {
    try {
      setGeneralChat(await readResponse<ChatMessage[]>(
        await fetch(`${API}/chat?session_id=${sessionId}`)));
    } catch {
      setGeneralChat([]);
    }
  };
  const refreshChatSessions = async () => {
    try {
      setChatSessions(await readResponse<ChatSession[]>(await fetch(`${API}/chat/sessions`)));
    } catch {
      /* 列表刷新失败不影响当前对话 */
    }
  };
  const ensureGeneralSession = async (): Promise<number> => {
    if (activeSessionId != null) return activeSessionId;
    const created = await readResponse<ChatSession>(
      await fetch(`${API}/chat/sessions`, { method: "POST" }));
    setChatSessions((rows) => [created, ...rows]);
    setActiveSessionId(created.id);
    return created.id;
  };
  const newGeneralSession = async () => {
    if (busy) return;
    try {
      const created = await readResponse<ChatSession>(
        await fetch(`${API}/chat/sessions`, { method: "POST" }));
      setChatSessions((rows) => [created, ...rows]);
      setActiveSessionId(created.id);
      setGeneralChat([]);
      setGeneralQuestion("");
    } catch (error) {
      setNotice(`新建会话失败：${reasonOf(error)}`);
    }
  };
  const selectGeneralSession = async (sessionId: number) => {
    if (busy || sessionId === activeSessionId) return;
    setActiveSessionId(sessionId);
    setGeneralQuestion("");
    await loadGeneralChat(sessionId);
  };
  const renameGeneralSession = async (sessionId: number, title: string) => {
    const cleaned = title.trim().slice(0, 80);
    if (!cleaned || busy) return;
    try {
      const updated = await readResponse<ChatSession>(await fetch(
        `${API}/chat/sessions/${sessionId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: cleaned }),
        }));
      setChatSessions((rows) => rows.map(
        (row) => (row.id === sessionId ? { ...row, ...updated } : row)));
    } catch (error) {
      setNotice(`重命名失败：${reasonOf(error)}`);
    }
  };
  const deleteGeneralSession = async (sessionId: number) => {
    if (busy) return;
    try {
      await readResponse(await fetch(`${API}/chat/sessions/${sessionId}`, { method: "DELETE" }));
      const remaining = chatSessions.filter((row) => row.id !== sessionId);
      setChatSessions(remaining);
      if (activeSessionId === sessionId) {
        const next = remaining[0]?.id ?? null;
        setActiveSessionId(next);
        setGeneralQuestion("");
        if (next != null) await loadGeneralChat(next);
        else setGeneralChat([]);
      }
      setNotice("会话已删除。");
    } catch (error) {
      setNotice(`删除会话失败：${reasonOf(error)}`);
    }
  };

  useEffect(() => {
    void loadPapers();
    void loadSettings();
    void (async () => {
      const rows = await loadChatSessions();
      const latest = rows[0]?.id ?? null;
      setActiveSessionId(latest);
      if (latest != null) await loadGeneralChat(latest);
    })();
  }, []);

  useEffect(() => {
    if (!papers.some((paper) => paper.status === "processing")) return;
    const timer = window.setInterval(() => void loadPapers(), 1500);
    return () => window.clearInterval(timer);
  }, [papers]);

  useEffect(() => {
    if (!selected || selected.paper.status !== "processing") return;
    const paperId = selected.paper.id;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const status = await readResponse<PaperStatus>(await fetch(`${API}/papers/${paperId}/status`));
          if (status.status !== "processing") {
            setSelected(await readResponse<PaperDetails>(await fetch(`${API}/papers/${paperId}`)));
            await loadPapers();
          }
        } catch {
          /* 下一页轮询继续 */
        }
      })();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [selected?.paper.id, selected?.paper.status]);

  const fetchStructured = async (paperId: number, refresh = false) => {
    setInsightsLoading(true);
    try {
      const response = refresh
        ? await fetch(`${API}/papers/${paperId}/structured`, { method: "POST" })
        : await fetch(`${API}/papers/${paperId}/structured`);
      if (!response.ok) throw new Error(await readError(response));
      setStructured(await readResponse<Structured>(response));
      if (refresh) setNotice("已刷新结构化精读结果。");
    } catch (error) {
      setNotice(`结构化精读失败：${reasonOf(error)}`);
    } finally {
      setInsightsLoading(false);
    }
  };

  const selectPaper = async (id: number) => {
    try {
      setSelected(await readResponse<PaperDetails>(await fetch(`${API}/papers/${id}`)));
      setTab("reader");
      setReaderTab("read");
      setResults([]);
      setChat([]);
      setHighlight(null);
      setStructured(null);
      const history = await fetch(`${API}/papers/${id}/chat`);
      if (history.ok) setChat(await history.json());
      void fetchStructured(id);
    } catch (error) {
      setNotice(`读取论文失败：${reasonOf(error)}`);
    }
  };

  const waitForPaper = async (id: number, openWhenReady: boolean): Promise<PaperStatus | null> => {
    for (let attempt = 0; attempt < 450; attempt += 1) {
      let status: PaperStatus;
      try {
        status = await readResponse<PaperStatus>(await fetch(`${API}/papers/${id}/status`));
      } catch {
        return null;
      }
      const active = status.status === "processing"
        || ["queued", "starting", "parsing", "embedding"].includes(status.progress_stage || "");
      if (!active) {
        await loadPapers();
        if (openWhenReady && status.status === "ready") {
          await selectPaper(id);
          setNotice("论文解析与本地索引已完成。");
        } else if (status.status === "failed") {
          setNotice(`处理失败：${status.error_message || "未知错误"}`);
        } else {
          setNotice(status.error_message ? `处理完成（${status.error_message}）` : "处理完成。");
          if (selected?.paper.id === id) {
            setSelected(await readResponse<PaperDetails>(await fetch(`${API}/papers/${id}`)));
          }
        }
        return status;
      }
      const counter = status.progress_total ? ` ${status.progress_current}/${status.progress_total}` : "";
      setNotice(`正在${stageLabel(status.progress_stage)}${counter}…`);
      await sleep(800);
    }
    setNotice("处理时间较长，可稍后在论文库中查看状态。");
    return null;
  };

  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setImporting(true);
    const body = new FormData();
    body.append("file", file);
    try {
      const payload = await readResponse<{ id: number; title: string }>(
        await fetch(`${API}/papers/import`, { method: "POST", body }),
      );
      setNotice(`“${payload.title}”已上传，正在本机解析…`);
      await loadPapers();
      await waitForPaper(payload.id, true);
    } catch (error) {
      setNotice(`导入失败：${reasonOf(error)}`);
    } finally {
      setImporting(false);
      event.target.value = "";
    }
  };

  const reindex = async () => {
    if (!selected) return;
    const paperId = selected.paper.id;
    setImporting(true);
    try {
      await readResponse(await fetch(`${API}/papers/${paperId}/reindex`, { method: "POST" }));
      setNotice("正在重建本地向量索引…");
      await waitForPaper(paperId, false);
    } catch (error) {
      setNotice(`重建索引失败：${reasonOf(error)}`);
    } finally {
      setImporting(false);
    }
  };

  const jumpTo: JumpHandler = (page, paperId, bbox) => {
    void (async () => {
      if (paperId != null && (!selected || selected.paper.id !== paperId)) {
        await selectPaper(paperId);
      }
      setHighlight({ page, bbox: bbox ?? null });
      setReaderTab("read");
      setTab("reader");
    })();
  };

  const search = async () => {
    if (!selected || !query.trim()) {
      setResults([]);
      return;
    }
    const endpoint = searchMode === "keyword" ? "search" : "retrieve";
    try {
      setResults(await readResponse<SearchResult[]>(
        await fetch(`${API}/papers/${selected.paper.id}/${endpoint}?q=${encodeURIComponent(query)}`),
      ));
    } catch (error) {
      setNotice(`检索失败：${reasonOf(error)}`);
    }
  };

  const searchLibrary = async () => {
    const value = libraryQuery.trim();
    if (!value) {
      setLibraryResults([]);
      return;
    }
    setLibrarySearching(true);
    try {
      setLibraryResults(await readResponse<SearchResult[]>(
        await fetch(`${API}/library/search?q=${encodeURIComponent(value)}&top_k=10`),
      ));
    } catch (error) {
      setNotice(`全库检索失败：${reasonOf(error)}`);
    } finally {
      setLibrarySearching(false);
    }
  };

  const patchLast = (setter: Dispatch<SetStateAction<ChatMessage[]>>,
                     update: (message: ChatMessage) => ChatMessage) => {
    setter((messages) => messages.map((message, index) =>
      index === messages.length - 1 ? update(message) : message));
  };

  const runAgent = async (
    mode: "paper" | "library" | "general",
    content: string,
    retry: boolean,
    paperId: number | undefined,
    setMessages: Dispatch<SetStateAction<ChatMessage[]>>,
    sessionId?: number,
  ) => {
    agentAbort.current?.abort();
    const controller = new AbortController();
    agentAbort.current = controller;
    setMessages((messages) => [...messages, { role: "assistant", content: "", pending: true }]);
    setBusy(true);
    try {
      const response = await fetch(`${API}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content, mode, paper_id: paperId, stream: true, retry,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await readError(response));
      let finished = false;
      await consumeAgentStream(response, (event) => {
        if (event.type === "delta") {
          patchLast(setMessages, (message) => ({
            ...message,
            content: message.content + event.text,
          }));
        } else if (event.type === "step") {
          patchLast(setMessages, (message) => ({
            ...message,
            trace: [...(message.trace || []), event],
          }));
        } else if (event.type === "done") {
          finished = true;
          patchLast(setMessages, (message) => ({
            ...message,
            id: event.id,
            content: event.content || message.content || "（模型没有返回内容）",
            citations: event.citations,
            sources: event.sources,
            trace: event.trace || message.trace,
            verified: event.verified,
            warnings: event.warnings,
            pending: false,
          }));
        } else if (event.type === "error") {
          finished = true;
          patchLast(setMessages, (message) => ({
            ...message,
            content: message.content || `⚠ ${event.error}`,
            error: event.error,
            pending: false,
          }));
        }
      });
      if (!finished) {
        patchLast(setMessages, (message) => ({
          ...message,
          content: message.content || "⚠ 流式连接提前结束，请重试。",
          error: message.error || "流式连接提前结束",
          pending: false,
        }));
      }
    } catch (error) {
      if (controller.signal.aborted) {
        patchLast(setMessages, (message) => ({
          ...message,
          content: message.content
            ? `${message.content}\n\n（已停止生成）`
            : "已停止生成。",
          pending: false,
        }));
      } else {
        const reason = reasonOf(error);
        patchLast(setMessages, (message) => ({
          ...message,
          content: message.content || `⚠ ${reason}`,
          error: reason,
          pending: false,
        }));
      }
    } finally {
      if (agentAbort.current === controller) agentAbort.current = null;
      setBusy(false);
    }
  };

  const stopAgent = () => {
    agentAbort.current?.abort();
  };

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    if (!selected || !question.trim()) return;
    const content = question;
    setChat((messages) => [...messages, { role: "user", content }]);
    setQuestion("");
    await runAgent("paper", content, false, selected.paper.id, setChat);
  };

  const retryChat = async () => {
    if (!selected) return;
    const lastUser = chat.map((message) => message.role).lastIndexOf("user");
    if (lastUser < 0) return;
    const content = chat[lastUser].content;
    setChat((messages) => messages.slice(0, lastUser + 1));
    await runAgent("paper", content, true, selected.paper.id, setChat);
  };

  const askGeneral = async (event: FormEvent) => {
    event.preventDefault();
    if (!generalQuestion.trim() || busy) return;
    const content = generalQuestion;
    const mode: GeneralMode = papers.length > 0 ? generalMode : "general";
    try {
      const sessionId = await ensureGeneralSession();
      setGeneralQuestion("");
      setGeneralChat((messages) => [...messages, { role: "user", content }]);
      await runAgent(mode, content, false, undefined, setGeneralChat, sessionId);
    } catch (error) {
      setNotice(`提问失败：${reasonOf(error)}`);
    } finally {
      await refreshChatSessions();
    }
  };

  const retryGeneral = async () => {
    const lastUser = generalChat.map((message) => message.role).lastIndexOf("user");
    if (lastUser < 0) return;
    const content = generalChat[lastUser].content;
    const mode: GeneralMode = papers.length > 0 ? generalMode : "general";
    setGeneralChat((messages) => messages.slice(0, lastUser + 1));
    try {
      const sessionId = await ensureGeneralSession();
      await runAgent(mode, content, true, undefined, setGeneralChat, sessionId);
    } catch (error) {
      setNotice(`重试失败：${reasonOf(error)}`);
    } finally {
      await refreshChatSessions();
    }
  };

  const remove = async () => {
    if (!selected || !window.confirm("删除论文、解析文本、索引、问答和本地 PDF？")) return;
    try {
      await readResponse(await fetch(`${API}/papers/${selected.paper.id}`, { method: "DELETE" }));
      setSelected(null);
      setTab("library");
      await loadPapers();
      setNotice("论文及其本地关联数据已删除。");
    } catch (error) {
      setNotice(`删除失败：${reasonOf(error)}`);
    }
  };

  const saveSettings = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      setSettings(await readResponse<Settings>(await fetch(`${API}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: settings.provider,
          protocol: settings.protocol,
          base_url: settings.base_url,
          model: settings.model,
          embedding_model: settings.embedding_model,
          ocr_enabled: settings.ocr_enabled,
          ocr_engine: settings.ocr_engine,
          ocr_lang: settings.ocr_lang,
          ocr_dpi: settings.ocr_dpi,
          ocr_endpoint: settings.ocr_endpoint,
          ocr_type: settings.ocr_type,
          ocr_access_key_id: settings.ocr_access_key_id,
          ocr_access_key_secret: ocrAccessKeySecret,
          rerank_enabled: settings.rerank_enabled,
          rerank_engine: settings.rerank_engine,
          rerank_model: settings.rerank_model,
          rerank_base_url: settings.rerank_base_url,
          relevance_min_coverage: settings.relevance_min_coverage,
          nli_enabled: settings.nli_enabled,
          nli_engine: settings.nli_engine,
          nli_model: settings.nli_model,
          context_token_budget: settings.context_token_budget,
          rerank_api_key: rerankApiKey,
          api_key: apiKey,
          embedding_api_key: embeddingApiKey,
        }),
      })));
      setApiKey("");
      setEmbeddingApiKey("");
      setRerankApiKey("");
      setOcrAccessKeySecret("");
      setNotice("模型设置已保存。API Key 仅保存到本机安全存储，不写入日志或数据库。");
    } catch (error) {
      setNotice(`设置保存失败：${reasonOf(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const loadModels = async () => {
    if (!settings.provider) {
      setNotice("请先选择模型供应商。");
      return;
    }
    try {
      const payload = await readResponse<{ models: string[] }>(await fetch(`${API}/models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: settings.provider,
          protocol: settings.protocol || undefined,
          base_url: settings.base_url || undefined,
          api_key: apiKey || undefined,
        }),
      }));
      setAvailableModels(payload.models);
      setNotice(`已读取 ${payload.models.length} 个可用模型。`);
    } catch (error) {
      setNotice(`读取模型列表失败：${reasonOf(error)}`);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">觅</span>
          <div>
            <span className="eyebrow">HEARTSEEKER_MED</span>
            <h1>觅心 · 可追溯的医学论文精读</h1>
          </div>
        </div>
        <label className="import-button">
          <input type="file" accept=".pdf,application/pdf" onChange={importFile} disabled={importing} />
          {importing ? "处理中…" : "导入 PDF"}
        </label>
      </header>
      {notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice("")}>×</button></div>}
      <div className="layout">
        <nav className="nav">
          <button className={tab === "chat" ? "nav-active" : ""} onClick={() => setTab("chat")}>
            ◈ <span>主对话 · 通用医学</span>
          </button>
          <button className={tab === "library" ? "nav-active" : ""} onClick={() => setTab("library")}>
            ◫ <span>论文库 · 全库检索</span>
          </button>
          <button className={tab === "stats" ? "nav-active" : ""} onClick={() => setTab("stats")}>
            ▤ <span>统计数据</span>
          </button>
          <button className={tab === "logs" ? "nav-active" : ""} onClick={() => setTab("logs")}>
            ≋ <span>运行日志</span>
          </button>
          {selected && (
            <button className={tab === "reader" ? "nav-active" : ""} onClick={() => setTab("reader")}>
              ▣ <span>论文工作台</span>
            </button>
          )}
          <button className={tab === "settings" ? "nav-active" : ""} onClick={() => setTab("settings")}>
            ⚙ <span>设置与隐私</span>
          </button>
          <div className="nav-foot">本地优先<br /><small>数据不会自动上传</small></div>
        </nav>
        <main className="main">
          {tab === "chat" && (
            <GeneralChat
              chat={generalChat}
              question={generalQuestion}
              setQuestion={setGeneralQuestion}
              ask={askGeneral}
              busy={busy}
              onStop={stopAgent}
              onRetry={() => void retryGeneral()}
              mode={generalMode}
              setMode={setGeneralMode}
              hasPapers={papers.length > 0}
              onJump={jumpTo}
              sessions={chatSessions}
              activeSessionId={activeSessionId}
              onSelectSession={(id) => void selectGeneralSession(id)}
              onNewSession={() => void newGeneralSession()}
              onRenameSession={(id, title) => void renameGeneralSession(id, title)}
              onDeleteSession={(id) => void deleteGeneralSession(id)}
            />
          )}
          {tab === "library" && (
            <Library
              papers={papers}
              selected={selected}
              onSelect={(id) => void selectPaper(id)}
              onImport={importFile}
              importing={importing}
              libraryQuery={libraryQuery}
              setLibraryQuery={setLibraryQuery}
              libraryResults={libraryResults}
              librarySearching={librarySearching}
              onSearch={() => void searchLibrary()}
              onOpenResult={(result) => jumpTo(result.page_number, result.paper_id, bboxOf(result))}
            />
          )}
          {tab === "stats" && <StatsView />}
          {tab === "logs" && <LogsView />}
          {tab === "settings" && (
            <SettingsView
              settings={settings}
              apiKey={apiKey}
              setApiKey={setApiKey}
              embeddingApiKey={embeddingApiKey}
              setEmbeddingApiKey={setEmbeddingApiKey}
              rerankApiKey={rerankApiKey}
              setRerankApiKey={setRerankApiKey}
              ocrAccessKeySecret={ocrAccessKeySecret}
              setOcrAccessKeySecret={setOcrAccessKeySecret}
              setSettings={setSettings}
              onSave={saveSettings}
              availableModels={availableModels}
              loadModels={() => void loadModels()}
            />
          )}
          {tab === "reader" && selected && (
            <Reader
              details={selected}
              readerTab={readerTab}
              setReaderTab={setReaderTab}
              query={query}
              setQuery={setQuery}
              searchMode={searchMode}
              setSearchMode={setSearchMode}
              results={results}
              search={() => void search()}
              highlight={highlight}
              onJump={jumpTo}
              structured={structured}
              insightsLoading={insightsLoading}
              refreshStructured={() => void fetchStructured(selected.paper.id, true)}
              chat={chat}
              question={question}
              setQuestion={setQuestion}
              ask={ask}
              busy={busy}
              onStop={stopAgent}
              onRetry={() => void retryChat()}
              remove={() => void remove()}
              reindex={() => void reindex()}
              importing={importing}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function StatCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="stat-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function StatsView() {
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    try {
      setStats(await readResponse<UsageStats>(await fetch(`${API}/stats`)));
      setError("");
    } catch (reason) {
      setError(reasonOf(reason));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const number = (value?: number) => (value ?? 0).toLocaleString("zh-CN");
  const modeNames: Record<string, string> = {
    paper: "单篇论文",
    library: "全库检索",
    general: "通用医学",
  };
  const tokens = stats?.tokens;

  return (
    <section className="stats-view">
      <div className="hero">
        <div>
          <span className="eyebrow">USAGE &amp; TELEMETRY</span>
          <h2>统计数据</h2>
          <p>本地累计的 token、Agent 运行与工具调用；所有记录仅保存在本机数据库。</p>
        </div>
        <button className="secondary" onClick={() => void load()} disabled={loading}>
          {loading ? "刷新中…" : "刷新"}
        </button>
      </div>
      {error && <div className="notice"><span>{error}</span><button onClick={() => setError("")}>×</button></div>}
      <div className="stat-grid">
        <StatCard label="累计 Token" value={number(tokens?.total)}
          hint={`提示 ${number(tokens?.prompt)} · 生成 ${number(tokens?.completion)}`} />
        <StatCard label="今日 Token" value={number(tokens?.today_total)}
          hint={`今日 ${number(tokens?.today_calls)} 次模型调用`} />
        <StatCard label="模型调用" value={number(tokens?.calls)}
          hint={tokens?.estimated_calls
            ? `其中 ${number(tokens.estimated_calls)} 次为估算值`
            : "均由模型服务返回"} />
        <StatCard label="Agent 运行" value={number(stats?.runs.total)}
          hint={`今日 ${number(stats?.runs.today_runs)} 次 · 引用核验通过 ${number(stats?.runs.verified)} 次`} />
        <StatCard label="工具调用" value={number(stats?.tools.total)}
          hint={`平均每次运行 ${stats?.runs.avg_steps ?? 0} 步`} />
        <StatCard label="会话消息" value={number(stats?.sessions.messages)}
          hint={`${number(stats?.sessions.total)} 个会话`} />
      </div>
      <div className="stat-columns">
        <div className="stat-panel">
          <div className="section-heading"><h3>模型用量</h3><span>{stats?.models.length ?? 0} 个模型</span></div>
          {stats?.models.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>供应商</th><th>模型</th><th>调用</th><th>Token</th></tr>
                </thead>
                <tbody>
                  {stats.models.map((item) => (
                    <tr key={`${item.provider}/${item.model}`}>
                      <td>{item.provider || "—"}</td>
                      <td>{item.model || "—"}</td>
                      <td>{number(item.calls)}{item.estimated_calls ? `（≈${number(item.estimated_calls)}）` : ""}</td>
                      <td>{number(item.total_tokens)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="muted">还没有模型调用记录。</p>}
        </div>
        <div className="stat-panel">
          <div className="section-heading"><h3>工具调用</h3><span>共 {number(stats?.tools.total)} 次</span></div>
          {stats?.tools.by_tool.length ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th>工具</th><th>次数</th></tr></thead>
                <tbody>
                  {stats.tools.by_tool.map((item) => (
                    <tr key={item.tool}><td>{item.tool}</td><td>{number(item.count)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="muted">还没有工具调用记录。</p>}
        </div>
      </div>
      <div className="stat-panel">
        <div className="section-heading">
          <h3>运行模式与论文库</h3>
          <span>论文 {number(stats?.papers.total)} 篇 · 已向量化 {number(stats?.papers.vectorized)} · 待 OCR {number(stats?.papers.needs_ocr)}</span>
        </div>
        <div className="stat-chips">
          {stats?.runs.modes.length ? stats.runs.modes.map((item) => (
            <span key={item.mode} className="stat-chip">
              {modeNames[item.mode] || item.mode}<b>{number(item.count)}</b>
            </span>
          )) : <p className="muted">还没有 Agent 运行记录。</p>}
        </div>
      </div>
    </section>
  );
}

function LogRow({ log }: { log: LogEvent }) {
  const { seq, time, level, logger, message, event, ...fields } = log;
  const entries = Object.entries(fields);
  const clock = new Date(time);
  const format = (value: unknown) => {
    const text = typeof value === "string" ? value : (JSON.stringify(value) ?? String(value));
    return text.length > 120 ? `${text.slice(0, 120)}…` : text;
  };
  return (
    <article className={`log-row level-${level.toLowerCase()}`}>
      <div className="log-head">
        <time>{Number.isNaN(clock.getTime()) ? time : clock.toLocaleTimeString("zh-CN", { hour12: false })}</time>
        <span className="log-level">{level}</span>
        <b>{event || message}</b>
        <small>{logger}</small>
        <span className="log-seq">#{seq}</span>
      </div>
      {entries.length > 0 && (
        <div className="log-fields">
          {entries.map(([key, value]) => (
            <span key={key} className="log-field"><em>{key}</em>={format(value)}</span>
          ))}
        </div>
      )}
    </article>
  );
}

function LogsView() {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState("");
  const cursor = useRef(0);
  const pausedRef = useRef(false);
  const polling = useRef(false);
  const firstPauseEffect = useRef(true);
  const listRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const poll = async () => {
    if (pausedRef.current || polling.current) return;
    polling.current = true;
    try {
      const events = await readResponse<LogEvent[]>(
        await fetch(`${API}/logs?after=${cursor.current}&limit=200`));
      if (events.length) {
        cursor.current = events[events.length - 1].seq;
        setLogs((current) => [...current, ...events].slice(-500));
      }
      setError("");
    } catch (reason) {
      setError(reasonOf(reason));
    } finally {
      polling.current = false;
    }
  };

  useEffect(() => {
    void poll();
    const timer = window.setInterval(() => void poll(), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    pausedRef.current = paused;
    if (firstPauseEffect.current) {
      firstPauseEffect.current = false;
      return;
    }
    if (!paused) void poll();
  }, [paused]);

  useLayoutEffect(() => {
    const container = listRef.current;
    if (!container || !stickToBottom.current) return;
    container.scrollTop = container.scrollHeight;
  }, [logs]);

  const handleScroll = () => {
    const container = listRef.current;
    if (!container) return;
    stickToBottom.current =
      container.scrollHeight - container.scrollTop - container.clientHeight < 40;
  };

  return (
    <section className="logs-view">
      <div className="hero">
        <div>
          <span className="eyebrow">STRUCTURED RUNTIME LOG</span>
          <h2>运行日志</h2>
          <p>实时显示模型调用、Agent 工具步骤、论文导入等结构化事件；仅保留最近 1000 条。</p>
        </div>
        <div className="log-actions">
          <span className={`log-status ${error ? "off" : paused ? "paused" : "on"}`}>
            {error ? "连接中断" : paused ? "已暂停" : "实时监听中"}
          </span>
          <button className="secondary" onClick={() => setPaused((value) => !value)}>
            {paused ? "继续" : "暂停"}
          </button>
          <button className="secondary" onClick={() => { setLogs([]); stickToBottom.current = true; }}>
            清空视图
          </button>
        </div>
      </div>
      {error && <div className="notice"><span>{error}</span><button onClick={() => setError("")}>×</button></div>}
      <div className="log-list" ref={listRef} onScroll={handleScroll}>
        {logs.length ? logs.map((log) => <LogRow key={log.seq} log={log} />) : (
          <div className="empty-state compact">
            <h3>暂无日志</h3>
            <p>发起一次对话或导入论文后，这里会实时出现结构化事件。</p>
          </div>
        )}
      </div>
    </section>
  );
}

function GeneralChat({
  chat,
  question,
  setQuestion,
  ask,
  busy,
  onStop,
  onRetry,
  mode,
  setMode,
  hasPapers,
  onJump,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onRenameSession,
  onDeleteSession,
}: {
  chat: ChatMessage[];
  question: string;
  setQuestion: (value: string) => void;
  ask: (event: FormEvent) => void;
  busy: boolean;
  onStop: () => void;
  onRetry: () => void;
  mode: GeneralMode;
  setMode: (mode: GeneralMode) => void;
  hasPapers: boolean;
  onJump: JumpHandler;
  sessions: ChatSession[];
  activeSessionId: number | null;
  onSelectSession: (sessionId: number) => void;
  onNewSession: () => void;
  onRenameSession: (sessionId: number, title: string) => void;
  onDeleteSession: (sessionId: number) => void;
}) {
  const messagesRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // Entering the tab remounts this component, so the effect lands on the
  // newest message every time.  While reading, streaming updates only follow
  // the bottom when the user has not scrolled away.
  useLayoutEffect(() => {
    const container = messagesRef.current;
    if (!container || !stickToBottom.current) return;
    container.scrollTop = container.scrollHeight;
  }, [chat]);

  const handleMessagesScroll = () => {
    const container = messagesRef.current;
    if (!container) return;
    stickToBottom.current =
      container.scrollHeight - container.scrollTop - container.clientHeight < 40;
  };

  const submit = (event: FormEvent) => {
    stickToBottom.current = true;
    ask(event);
  };

  const activeSession = sessions.find((session) => session.id === activeSessionId) ?? null;
  const renameSession = () => {
    if (activeSessionId == null || busy) return;
    const title = window.prompt("重命名会话", activeSession?.title || "");
    if (title == null || !title.trim()) return;
    onRenameSession(activeSessionId, title);
  };
  const deleteSession = () => {
    if (activeSessionId == null || busy) return;
    const label = activeSession?.title || `未命名会话 #${activeSessionId}`;
    if (!window.confirm(`删除会话「${label}」及其全部消息？`)) return;
    onDeleteSession(activeSessionId);
  };

  return (
    <section className="general-chat">
      <div className="chat-intro">
        <span className="eyebrow">MEDICAL LEARNING ASSISTANT</span>
        <h3>AI 学习助手 · 通用医学对话</h3>
        <p>可连续追问。开启“论文库检索”时，助手会多步检索你的本地论文并核验引用；关闭后按通用医学知识回答。</p>
        <div className="mode-toggle">
          <button
            type="button"
            className={mode === "library" ? "active" : ""}
            disabled={!hasPapers}
            onClick={() => setMode("library")}
          >
            论文库检索
          </button>
          <button
            type="button"
            className={mode === "general" ? "active" : ""}
            onClick={() => setMode("general")}
          >
            通用医学知识
          </button>
          {!hasPapers && <small>导入论文后可启用库内检索</small>}
        </div>
      </div>
      <div className="session-bar">
        <label>
          <span className="eyebrow">会话记忆</span>
          <select
            value={activeSessionId ?? ""}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (value) onSelectSession(value);
            }}
            disabled={busy}
          >
            {activeSessionId == null && <option value="">新会话</option>}
            {sessions.map((session) => (
              <option key={session.id} value={session.id}>
                {session.title || `未命名会话 #${session.id}`}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={onNewSession} disabled={busy}>
          ＋ 新会话
        </button>
        <button type="button" onClick={renameSession}
          disabled={busy || activeSessionId == null}>
          重命名
        </button>
        <button type="button" className="danger" onClick={deleteSession}
          disabled={busy || activeSessionId == null}>
          删除会话
        </button>
        <small>每个会话独立保留上下文；首条提问会取前四十个字生成名称，之后可手动重命名。</small>
      </div>
      <div className="messages general-messages" ref={messagesRef} onScroll={handleMessagesScroll}>
        {chat.length ? chat.map((message, index) => (
          <MessageView key={message.id || index} message={message} onJump={onJump} onRetry={onRetry} />
        )) : (
          <div className="empty-state compact">
            <h3>开始提问</h3>
            <p>例如：急性心衰和慢性心衰的治疗原则有什么区别？</p>
          </div>
        )}
      </div>
      <form className="question-form" onSubmit={submit}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="问一个医学问题，可连续追问…"
        />
        {busy && (
          <button type="button" className="stop-button" onClick={onStop}>停止</button>
        )}
        <button disabled={busy}>发送</button>
      </form>
    </section>
  );
}

function Library({
  papers,
  selected,
  onSelect,
  onImport,
  importing,
  libraryQuery,
  setLibraryQuery,
  libraryResults,
  librarySearching,
  onSearch,
  onOpenResult,
}: {
  papers: Paper[];
  selected: PaperDetails | null;
  onSelect: (id: number) => void;
  onImport: (event: ChangeEvent<HTMLInputElement>) => void;
  importing: boolean;
  libraryQuery: string;
  setLibraryQuery: (value: string) => void;
  libraryResults: SearchResult[];
  librarySearching: boolean;
  onSearch: () => void;
  onOpenResult: (result: SearchResult) => void;
}) {
  return (
    <section className="library">
      <div className="hero">
        <div>
          <span className="eyebrow">LOCAL RESEARCH DESK</span>
          <h2>你的论文库</h2>
          <p>导入医学论文，在本机完成解析、混合检索与可核验阅读。</p>
        </div>
        <label className="primary">
          <input type="file" accept=".pdf,application/pdf" onChange={onImport} disabled={importing} />
          {importing ? "处理中…" : "＋ 导入论文"}
        </label>
      </div>
      <div className="privacy-card">
        <b>隐私边界</b>
        <span>PDF、解析文本、切片、向量和问答历史默认只保存在本机。只有主动使用模型功能时，才会发送问题和必要原文片段。</span>
      </div>
      <div className="library-search">
        <div>
          <span className="eyebrow">LIBRARY-WIDE HYBRID SEARCH</span>
          <h3>全库检索</h3>
        </div>
        <div className="library-search-row">
          <input
            value={libraryQuery}
            onChange={(event) => setLibraryQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onSearch()}
            placeholder="跨论文检索，如：SGLT2 inhibitor heart failure"
          />
          <button type="button" onClick={onSearch} disabled={librarySearching}>
            {librarySearching ? "检索中…" : "检索"}
          </button>
        </div>
        {libraryResults.length > 0 && (
          <div className="library-results">
            {libraryResults.map((result, index) => (
              <button
                type="button"
                className="library-result"
                key={`${result.id}-${index}`}
                onClick={() => onOpenResult(result)}
              >
                <span>
                  {result.paper_title || `论文 #${result.paper_id}`} · PDF 第 {result.page_number} 页
                  {result.section && ` · ${result.section}`}
                  {result.sources?.length ? ` · ${result.sources.join(" + ")}` : ""}
                </span>
                <p>{result.text.slice(0, 220)}</p>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="section-heading">
        <h3>全部论文</h3>
        <span>{papers.length} 篇</span>
      </div>
      {papers.length ? (
        <div className="paper-grid">
          {papers.map((paper) => {
            const total = paper.progress_total || 0;
            const current = paper.progress_current || 0;
            const ratio = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 35;
            return (
              <button
                className={`paper-card ${selected?.paper.id === paper.id ? "selected" : ""}`}
                key={paper.id}
                onClick={() => onSelect(paper.id)}
              >
                <div className="paper-icon">PDF</div>
                <div className="paper-info">
                  <h3>{paper.title}</h3>
                  <p>{paper.file_name}</p>
                  <small>
                    {paper.page_count} 页 · {paper.status === "ready"
                      ? (paper.needs_ocr ? "需 OCR" : paper.vectorized ? "已完成 · 混合索引" : "已完成 · 仅关键词")
                      : paper.status === "processing"
                        ? `${stageLabel(paper.progress_stage)}${total ? ` ${current}/${total}` : ""}`
                        : paper.status}
                  </small>
                  {paper.status === "processing" && (
                    <div className="progress-track"><span style={{ width: `${ratio}%` }} /></div>
                  )}
                </div>
                <span className="arrow">›</span>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-icon">＋</div>
          <h3>还没有论文</h3>
          <p>导入一篇医学 PDF，开始建立你的本地证据库。</p>
        </div>
      )}
    </section>
  );
}

function Reader({
  details,
  readerTab,
  setReaderTab,
  query,
  setQuery,
  searchMode,
  setSearchMode,
  results,
  search,
  highlight,
  onJump,
  structured,
  insightsLoading,
  refreshStructured,
  chat,
  question,
  setQuestion,
  ask,
  busy,
  onStop,
  onRetry,
  remove,
  reindex,
  importing,
}: {
  details: PaperDetails;
  readerTab: "read" | "insights" | "chat";
  setReaderTab: (tab: "read" | "insights" | "chat") => void;
  query: string;
  setQuery: (value: string) => void;
  searchMode: "hybrid" | "keyword";
  setSearchMode: (mode: "hybrid" | "keyword") => void;
  results: SearchResult[];
  search: () => void;
  highlight: Highlight | null;
  onJump: JumpHandler;
  structured: Structured | null;
  insightsLoading: boolean;
  refreshStructured: () => void;
  chat: ChatMessage[];
  question: string;
  setQuestion: (value: string) => void;
  ask: (event: FormEvent) => void;
  busy: boolean;
  onStop: () => void;
  onRetry: () => void;
  remove: () => void;
  reindex: () => void;
  importing: boolean;
}) {
  const paperId = details.paper.id;
  return (
    <section className="reader">
      <div className="reader-head">
        <div>
          <span className="eyebrow">论文工作台</span>
          <h2>{details.paper.title}</h2>
          <p>
            {details.paper.file_name} · {details.paper.page_count} 页
            {details.paper.embedding_model && ` · ${details.paper.embedding_model}`}
            {details.paper.embedding_dim ? ` (${details.paper.embedding_dim}d)` : ""}
          </p>
        </div>
        <div className="head-actions">
          <a href={`${API}/papers/${paperId}/file`} target="_blank" rel="noreferrer">打开 PDF</a>
          <button onClick={reindex} disabled={importing}>{importing ? "索引处理中…" : "重建索引"}</button>
          <button className="danger" onClick={remove}>删除</button>
        </div>
      </div>
      <div className="reader-tabs">
        <button className={readerTab === "read" ? "active" : ""} onClick={() => setReaderTab("read")}>原文阅读</button>
        <button className={readerTab === "insights" ? "active" : ""} onClick={() => setReaderTab("insights")}>结构化精读</button>
        <button className={readerTab === "chat" ? "active" : ""} onClick={() => setReaderTab("chat")}>论文内问答</button>
      </div>
      {readerTab === "read" && (
        <div className="reader-grid">
          <div className="pdf-column">
            <div className="searchbar">
              <select
                className="search-mode"
                value={searchMode}
                onChange={(event) => setSearchMode(event.target.value as "hybrid" | "keyword")}
              >
                <option value="hybrid">混合检索</option>
                <option value="keyword">关键词检索</option>
              </select>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && search()}
                placeholder="搜索原文，如 primary outcome"
              />
              <button onClick={search}>检索</button>
            </div>
            <PdfViewer fileUrl={`${API}/papers/${paperId}/file`} pageCount={details.paper.page_count} highlight={highlight} />
          </div>
          <Evidence
            results={results}
            onJump={(result) => onJump(result.page_number, result.paper_id ?? paperId, bboxOf(result))}
          />
        </div>
      )}
      {readerTab === "insights" && (
        <Insights
          structured={structured}
          loading={insightsLoading}
          refresh={refreshStructured}
          onJump={(page) => onJump(page, paperId)}
        />
      )}
      {readerTab === "chat" && (
        <Chat
          chat={chat}
          question={question}
          setQuestion={setQuestion}
          ask={ask}
          onJump={onJump}
          busy={busy}
          onStop={onStop}
          onRetry={onRetry}
        />
      )}
    </section>
  );
}

function PdfViewer({ fileUrl, pageCount, highlight }: { fileUrl: string; pageCount: number; highlight: Highlight | null }) {
  const [pdf, setPdf] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    const task = pdfjsLib.getDocument(fileUrl);
    task.promise.then((doc) => { if (!cancelled) setPdf(doc); }).catch((err) => { if (!cancelled) setError(err?.message || String(err)); });
    return () => { cancelled = true; };
  }, [fileUrl]);
  useEffect(() => {
    if (highlight) document.getElementById(`page-${highlight.page}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [highlight]);
  if (error) return <div className="pdf-error">无法加载 PDF：{error}</div>;
  if (!pdf) return <div className="pdf-loading">正在加载 PDF…</div>;
  const pageNumbers = Array.from({ length: pageCount }, (_, i) => i + 1);
  return (
    <div className="pdf-viewer">
      {pageNumbers.map((n) => (
        <PdfPage key={n} pdf={pdf} pageNumber={n} highlight={highlight && highlight.page === n ? highlight : null} />
      ))}
    </div>
  );
}

function PdfPage({ pdf, pageNumber, highlight }: { pdf: any; pageNumber: number; highlight: Highlight | null }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rendered, setRendered] = useState(false);
  const [baseViewport, setBaseViewport] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setRendered(true);
        observer.disconnect();
      }
    }, { rootMargin: "600px 0px" });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!rendered || !pdf) return;
    let cancelled = false;
    (async () => {
      try {
        const page = await pdf.getPage(pageNumber);
        const base = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({ scale: 1.5 });
        const canvas = canvasRef.current;
        if (!canvas || cancelled) return;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        await page.render({ canvasContext: ctx, viewport }).promise;
        if (!cancelled) setBaseViewport({ width: base.width, height: base.height });
      } catch (error) {
        console.error("渲染 PDF 页面失败", error);
      }
    })();
    return () => { cancelled = true; };
  }, [rendered, pdf, pageNumber]);

  const highlightStyle = highlight && highlight.bbox && baseViewport ? {
    left: `${(highlight.bbox[0] / baseViewport.width) * 100}%`,
    top: `${(highlight.bbox[1] / baseViewport.height) * 100}%`,
    width: `${((highlight.bbox[2] - highlight.bbox[0]) / baseViewport.width) * 100}%`,
    height: `${((highlight.bbox[3] - highlight.bbox[1]) / baseViewport.height) * 100}%`,
  } : undefined;

  return (
    <div className="pdf-page" id={`page-${pageNumber}`} ref={wrapRef}>
      <div className="pdf-page-label">PDF 第 {pageNumber} 页</div>
      <div className="pdf-canvas-wrap">
        {rendered
          ? <canvas ref={canvasRef} style={{ width: "100%", height: "auto", display: "block" }} />
          : <div className="pdf-page-placeholder">滚动加载…</div>}
        {highlightStyle && <div className="pdf-highlight" style={highlightStyle} />}
      </div>
    </div>
  );
}

function Evidence({ results, onJump }: { results: SearchResult[]; onJump: (result: SearchResult) => void }) {
  return (
    <aside className="evidence-panel">
      <div className="panel-title"><h3>引用证据</h3><span>{results.length} 条</span></div>
      {results.length ? results.map((result) => (
        <button className="evidence-item" key={result.id} onClick={() => onJump(result)}>
          <span>
            PDF 第 {result.page_number} 页
            {result.section && ` · ${result.section}`}
            {result.sources?.length ? ` · ${result.sources.join(" + ")}` : ""}
            {result.score != null && ` · RRF ${result.score.toFixed(4)}`}
          </span>
          <p>{result.text}</p>
          <small>切片 #{result.id} · 点击高亮原文</small>
        </button>
      )) : <p className="empty">检索结果会显示在这里，点击可在 PDF 上高亮对应位置。</p>}
    </aside>
  );
}

function Insights({
  structured,
  loading,
  refresh,
  onJump,
}: {
  structured: Structured | null;
  loading: boolean;
  refresh: () => void;
  onJump: (page: number) => void;
}) {
  const fields = structured?.fields || {};
  return (
    <div className="insights">
      <div className="insight-intro">
        <span className="eyebrow">EVIDENCE-GROUNDED SUMMARY</span>
        <h3>结构化论文精读</h3>
        <p>每个字段都附带可核验的原文引文；没有明确原文证据的字段会标记为“论文未明确报告”。</p>
        <div className="insight-actions">
          {structured && (
            <span className="insight-source">
              {structured.source === "llm" ? "LLM 提取" : "规则提取"}
              {structured.cached ? " · 缓存" : ""}
              {structured.llm_available === false ? " · 未配置模型 Key" : ""}
            </span>
          )}
          <button type="button" onClick={refresh} disabled={loading}>
            {loading ? "提取中…" : "刷新精读"}
          </button>
        </div>
        {structured?.warnings?.map((warning, index) => (
          <p className="insight-warning" key={`${warning}-${index}`}>⚠ {warning}</p>
        ))}
      </div>
      <div className="insight-grid">
        {Object.entries(fields).map(([key, field]) => (
          <article className="insight-card" key={key}>
            <span>{field.label || key}</span>
            <strong>{field.value || "论文未明确报告"}</strong>
            {field.evidence?.map((evidence, index) => (
              <button
                className="source-chip"
                key={`${evidence.page_number}-${index}`}
                onClick={() => onJump(evidence.page_number)}
              >
                PDF 第 {evidence.page_number} 页
                {evidence.quote ? ` · ${evidence.quote.slice(0, 120)}` : ""}
              </button>
            ))}
            {field.verified === false && <small className="unverified-text">未找到可核验原文</small>}
          </article>
        ))}
      </div>
      {!structured && <div className="empty-state compact"><h3>正在读取结构化信息…</h3></div>}
    </div>
  );
}

function Chat({
  chat,
  question,
  setQuestion,
  ask,
  onJump,
  busy,
  onStop,
  onRetry,
}: {
  chat: ChatMessage[];
  question: string;
  setQuestion: (value: string) => void;
  ask: (event: FormEvent) => void;
  onJump: JumpHandler;
  busy: boolean;
  onStop: () => void;
  onRetry: () => void;
}) {
  return (
    <div className="chat">
      <div className="chat-intro">
        <span className="eyebrow">AGENTIC ONE-PAPER RAG</span>
        <h3>只基于当前论文提问</h3>
        <p>助手会先检索、必要时翻阅页面或章节，再生成带原文引文的回答；引用会逐字核对。</p>
      </div>
      <div className="messages">
        {chat.length ? chat.map((message, index) => (
          <MessageView key={message.id || index} message={message} onJump={onJump} onRetry={onRetry} />
        )) : (
          <div className="empty-state compact">
            <h3>还没有问题</h3>
            <p>例如：这项研究的主要终点是什么？</p>
          </div>
        )}
      </div>
      <form className="question-form" onSubmit={ask}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="输入关于当前论文的问题…"
        />
        {busy && (
          <button type="button" className="stop-button" onClick={onStop}>停止</button>
        )}
        <button disabled={busy}>发送</button>
      </form>
    </div>
  );
}

function SettingsView({
  settings,
  apiKey,
  setApiKey,
  embeddingApiKey,
  setEmbeddingApiKey,
  rerankApiKey,
  setRerankApiKey,
  ocrAccessKeySecret,
  setOcrAccessKeySecret,
  setSettings,
  onSave,
  availableModels,
  loadModels,
}: {
  settings: Settings;
  apiKey: string;
  setApiKey: (value: string) => void;
  embeddingApiKey: string;
  setEmbeddingApiKey: (value: string) => void;
  rerankApiKey: string;
  setRerankApiKey: (value: string) => void;
  ocrAccessKeySecret: string;
  setOcrAccessKeySecret: (value: string) => void;
  setSettings: (value: Settings) => void;
  onSave: (event: FormEvent) => void;
  availableModels: string[];
  loadModels: () => void;
}) {
  const [ocrTesting, setOcrTesting] = useState(false);
  const [ocrTestResult, setOcrTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const protocolLabel = settings.protocols.find((item) => item.id === settings.protocol)?.label
    || settings.protocol
    || "未配置";

  const testAliyunOcr = async () => {
    setOcrTesting(true);
    setOcrTestResult(null);
    try {
      const result = await readResponse<{ text: string; elapsed_ms: number }>(
        await fetch(`${API}/ocr/test`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            access_key_id: settings.ocr_access_key_id,
            access_key_secret: ocrAccessKeySecret,
            endpoint: settings.ocr_endpoint,
            ocr_type: settings.ocr_type,
          }),
        }),
      );
      setOcrTestResult({
        ok: true,
        message: `识别成功（${result.elapsed_ms} ms）：${result.text}`,
      });
    } catch (error) {
      setOcrTestResult({ ok: false, message: `连接失败：${reasonOf(error)}` });
    } finally {
      setOcrTesting(false);
    }
  };

  return (
    <section className="settings">
      <span className="eyebrow">LOCAL CONFIGURATION</span>
      <h2>设置与隐私</h2>
      <p className="subtitle">模型服务只在你主动发起精读或问答时使用。</p>
      <form onSubmit={onSave} className="settings-form">
        <label>
          模型供应商
          <select
            value={settings.provider}
            onChange={(event) => {
              const provider = event.target.value;
              const info = settings.providers.find((item) => item.id === provider);
              setSettings({
                ...settings,
                provider,
                provider_label: info?.label || "",
                protocol: info?.protocol || "",
                default_protocol: info?.protocol || "",
                base_url: info?.base_url || "",
                default_base_url: info?.base_url || "",
                model: info?.default_model || "",
                default_model: info?.default_model || "",
                api_key_configured: false,
              });
            }}
          >
            <option value="">未配置</option>
            {settings.providers.map((item) => (
              <option key={item.id} value={item.id}>{item.label}</option>
            ))}
          </select>
        </label>
        <label>
          接口协议
          {settings.provider === "custom" ? (
            <select
              value={settings.protocol || "openai"}
              onChange={(event) => setSettings({ ...settings, protocol: event.target.value })}
            >
              {settings.protocols.map((item) => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
          ) : (
            <input value={protocolLabel} readOnly />
          )}
          <small className="field-help">
            供应商按协议适配：OpenAI 兼容、Anthropic Messages、Google Gemini。自定义供应商可任选一种。
          </small>
        </label>
        <label>
          Base URL
          <input
            value={settings.base_url}
            onChange={(event) => setSettings({ ...settings, base_url: event.target.value })}
            placeholder={settings.default_base_url || "例如 https://api.openai.com/v1"}
          />
          <small className="field-help">
            通常填写到版本路径（如 /v1）；留空使用该供应商默认地址。自定义供应商必须填写。
          </small>
        </label>
        <label>
          API Key
          <input
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder={settings.api_key_configured
              ? "已配置（输入新 Key 可替换）"
              : "输入后可读取模型列表"}
          />
        </label>
        <div className="model-row">
          <select
            value={availableModels.includes(settings.model) ? settings.model : ""}
            onChange={(event) => setSettings({ ...settings, model: event.target.value })}
          >
            <option value="">选择已读取的模型</option>
            {availableModels.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>
          <button type="button" onClick={loadModels}>读取可用模型</button>
        </div>
        <label>
          模型名称
          <input
            value={settings.model}
            onChange={(event) => setSettings({ ...settings, model: event.target.value })}
            placeholder="也可手动输入模型 ID"
          />
          <small className="field-help">读取列表后可直接选择；保存后问答请求会使用这里的名称。</small>
        </label>
        <div className="security-note">
          读取模型列表和远端问答都会向所选供应商发送请求。API Key 不会发送给开发者服务器，也不会写入论文数据库。
        </div>
        <h3 className="settings-section">语义检索嵌入</h3>
        <label>
          嵌入模型
          <input
            value={settings.embedding_model}
            onChange={(event) => setSettings({ ...settings, embedding_model: event.target.value })}
            placeholder="text-embedding-v3"
          />
          <small className="field-help">
            固定使用千问 DashScope 兼容接口（默认 text-embedding-v3），向量仍存本机。
          </small>
        </label>
        <label>
          DashScope 嵌入 API Key
          <input
            type="password"
            value={embeddingApiKey}
            onChange={(event) => setEmbeddingApiKey(event.target.value)}
            placeholder={settings.embedding_key_configured
              ? "已配置（输入新 Key 可替换）"
              : "可与问答供应商的 Key 不同"}
          />
          <small className="field-help">
            嵌入 Key 与问答 Key 分开保存；问答供应商不是 Qwen 时仍需单独配置。
          </small>
        </label>
        <h3 className="settings-section">本地处理管线</h3>
        <div className="security-note">
          <b>OCR 什么时候会被调用：</b>只在导入 PDF（或对失败论文点“重试”）时逐页判断，
          检索、问答、重建索引都不会触发。一页同时满足三个条件才会识别：
          ① 已启用 OCR；② 所选引擎可用；③ 该页文本层少于 20 个字符。
          所以扫描件、纯图片页、几乎没有文字层的页面会走 OCR，
          正常带文字层的页面（≥20 字符）不会。单页识别失败不阻断导入；
          整篇没有任何文本时，论文会标记为“需 OCR”。
        </div>
        <label>
          扫描件 OCR
          <select
            value={settings.ocr_enabled || "true"}
            onChange={(event) => setSettings({ ...settings, ocr_enabled: event.target.value })}
          >
            <option value="true">启用</option>
            <option value="false">关闭</option>
          </select>
          <small className="field-help">
            当前可用引擎：{settings.ocr_available || "未检测"}。未安装/未配置引擎时扫描页会标记为 needs_ocr，不影响文字版 PDF。
          </small>
        </label>
        <label>
          OCR 引擎
          <select
            value={settings.ocr_engine || "auto"}
            onChange={(event) => setSettings({ ...settings, ocr_engine: event.target.value })}
          >
            <option value="auto">自动（RapidOCR → Tesseract）</option>
            <option value="rapidocr">RapidOCR（本地）</option>
            <option value="tesseract">Tesseract（本地）</option>
            <option value="aliyun">阿里云 OCR（RecognizeAllText）</option>
            <option value="none">不启用</option>
          </select>
          <small className="field-help">
            自动模式按 RapidOCR → Tesseract 顺序选择，两者都不可用时视为无引擎。
            阿里云 OCR 会把扫描页图像上传到阿里云，按调用量计费；本地方案不产生网络请求。
          </small>
        </label>
        <label>
          OCR 语言
          <input
            value={settings.ocr_lang || ""}
            onChange={(event) => setSettings({ ...settings, ocr_lang: event.target.value })}
            placeholder="chi_sim+eng"
          />
          <small className="field-help">仅本地 RapidOCR/Tesseract 使用；阿里云 OCR 由识别类型决定语言。</small>
        </label>
        <label>
          OCR 渲染 DPI
          <input
            type="number"
            min={72}
            max={600}
            value={settings.ocr_dpi || "220"}
            onChange={(event) => setSettings({ ...settings, ocr_dpi: event.target.value })}
          />
          <small className="field-help">仅影响扫描页识别，DPI 越高越慢；建议 200–300。</small>
        </label>
        {settings.ocr_engine === "aliyun" && (
          <>
            <h3 className="settings-section">阿里云 OCR 配置</h3>
            <div className="security-note">
              <b>说明：</b>以下配置只保存在本机。AccessKey Secret 写入系统凭据库（keyring），
              AccessKey ID、Endpoint 和识别类型写入本地设置数据库，均不会发送到除阿里云 OCR 之外的服务。
              当前状态：{settings.ocr_access_key_id ? "AccessKey ID 已保存" : "AccessKey ID 未填写"}；
              {settings.ocr_key_configured ? " Secret 已保存" : " Secret 未保存"}
              {settings.ocr_env_configured ? "；检测到 AccessKey 环境变量（设置页留空时使用）" : ""}。
            </div>
            <label>
              阿里云 AccessKey ID
              <input
                value={settings.ocr_access_key_id || ""}
                onChange={(event) => setSettings({ ...settings, ocr_access_key_id: event.target.value })}
                placeholder="LTAI..."
              />
              <small className="field-help">
                阿里云控制台 → 访问控制 RAM → 用户 → 创建 AccessKey。建议使用 RAM 子账号，仅授予
                {" "}<code>ocr:RecognizeAllText</code>（Resource 为 <code>*</code>），不要使用主账号 AccessKey。
              </small>
            </label>
            <label>
              阿里云 AccessKey Secret
              <input
                type="password"
                value={ocrAccessKeySecret}
                onChange={(event) => setOcrAccessKeySecret(event.target.value)}
                placeholder={settings.ocr_key_configured
                  ? "已配置（输入新 Secret 可替换）"
                  : "仅保存在系统凭据库"}
              />
              <small className="field-help">
                保存后只会显示“已配置”，不会回显；系统凭据库不可用时请改用环境变量
                {" "}<code>ALIBABA_CLOUD_ACCESS_KEY_ID</code> / <code>ALIBABA_CLOUD_ACCESS_KEY_SECRET</code>。
              </small>
            </label>
            <label>
              阿里云 OCR Endpoint
              <input
                list="aliyun-ocr-endpoints"
                value={settings.ocr_endpoint || ""}
                onChange={(event) => setSettings({ ...settings, ocr_endpoint: event.target.value })}
                placeholder="ocr-api.cn-hangzhou.aliyuncs.com"
              />
              <datalist id="aliyun-ocr-endpoints">
                <option value="ocr-api.cn-hangzhou.aliyuncs.com" />
                <option value="ocr-api-vpc.cn-hangzhou.aliyuncs.com" />
                <option value="ocr-api.cn-shanghai.aliyuncs.com" />
                <option value="ocr-api-vpc.cn-shanghai.aliyuncs.com" />
                <option value="ocr-api.cn-beijing.aliyuncs.com" />
                <option value="ocr-api-vpc.cn-beijing.aliyuncs.com" />
                <option value="ocr-api.cn-shenzhen.aliyuncs.com" />
                <option value="ocr-api-vpc.cn-shenzhen.aliyuncs.com" />
              </datalist>
              <small className="field-help">
                公网：<code>ocr-api.&lt;地域&gt;.aliyuncs.com</code>；同地域 VPC：
                {" "}<code>ocr-api-vpc.&lt;地域&gt;.aliyuncs.com</code>。保存时自动去掉
                {" "}<code>https://</code> 前缀和末尾斜杠。
              </small>
            </label>
            <label>
              阿里云 OCR 类型
              <select
                value={settings.ocr_type || "Advanced"}
                onChange={(event) => setSettings({ ...settings, ocr_type: event.target.value })}
              >
                <option value="General">General（通用高精度，适合普通文档）</option>
                <option value="Advanced">Advanced（高级识别，默认，适合论文版面）</option>
                <option value="MultiLang">MultiLang（多语言混合文档）</option>
              </select>
              <small className="field-help">
                识别类型决定阿里云使用的模型：普通论文建议 Advanced；中英混排或多语种文献可选 MultiLang。
              </small>
            </label>
            <div className="security-note">
              <b>图像限制：</b>支持 PDF、PNG、JPG/JPEG、BMP、GIF、TIFF、WebP；单张图像小于 10MB，
              宽高在 15–8192 像素之间，长宽比小于 50，建议大于 500 像素。本应用按页渲染 PNG，
              超过 10MB 时自动降低 DPI；仍超限则该页记录 OCR 失败。<br />
              <b>计费与隐私：</b>扫描页图像会上传至阿里云 OCR 并按调用量计费；文字版 PDF 和本地
              RapidOCR/Tesseract 方案不会产生该请求。
            </div>
            <div className="field-help">
              <button
                type="button"
                className="primary"
                onClick={() => void testAliyunOcr()}
                disabled={ocrTesting}
              >
                {ocrTesting ? "正在测试…" : "测试阿里云 OCR 连接"}
              </button>
              <small
                style={{
                  display: "block",
                  marginTop: 8,
                  color: ocrTestResult ? (ocrTestResult.ok ? "#176b57" : "#a34040") : undefined,
                }}
              >
                {ocrTestResult
                  ? ocrTestResult.message
                  : "使用当前填写的凭据和 Endpoint 识别一张测试图片；会消耗一次 OCR 调用。"}
              </small>
            </div>
          </>
        )}
        <label>
          检索重排
          <select
            value={settings.rerank_enabled || "true"}
            onChange={(event) => setSettings({ ...settings, rerank_enabled: event.target.value })}
          >
            <option value="true">启用</option>
            <option value="false">关闭</option>
          </select>
        </label>
        <label>
          重排引擎
          <select
            value={settings.rerank_engine || "lexical"}
            onChange={(event) => setSettings({ ...settings, rerank_engine: event.target.value })}
          >
            <option value="lexical">本地词法（无需模型）</option>
            <option value="qwen">Qwen 远程重排（Qwen3.7）</option>
            <option value="cross-encoder">本地 cross-encoder</option>
            <option value="none">不重排</option>
          </select>
        </label>
        <label>
          重排模型
          <input
            value={settings.rerank_model || ""}
            onChange={(event) => setSettings({ ...settings, rerank_model: event.target.value })}
            placeholder="qwen3.7-text-rerank 或 BAAI/bge-reranker-v2-m3"
          />
          <small className="field-help">
            Qwen 重排留空时默认 qwen3.7-text-rerank；本地 cross-encoder 需填写模型名。模型加载或调用失败会自动退回本地词法重排。
          </small>
        </label>
        <label>
          重排 Base URL
          <input
            value={settings.rerank_base_url || ""}
            onChange={(event) => setSettings({ ...settings, rerank_base_url: event.target.value })}
            placeholder="https://.../api/v1/services/rerank/text-rerank/text-rerank"
          />
          <small className="field-help">仅 Qwen 远程重排使用；留空使用默认地址。</small>
        </label>
        <label>
          重排 API Key
          <input
            type="password"
            value={rerankApiKey}
            onChange={(event) => setRerankApiKey(event.target.value)}
            placeholder={settings.rerank_key_configured
              ? "已配置（输入新 Key 可替换）"
              : "Qwen 重排服务的 Bearer Key"}
          />
          <small className="field-help">
            单独保存在系统凭据库；留空时依次尝试千问嵌入 Key 和 DASHSCOPE_API_KEY。
          </small>
        </label>
        <label>
          重排最低词覆盖
          <input
            value={settings.relevance_min_coverage || "0"}
            onChange={(event) => setSettings({ ...settings, relevance_min_coverage: event.target.value })}
            placeholder="0"
          />
          <small className="field-help">0–1；高于该覆盖率的片段才保留，0 表示不过滤。</small>
        </label>
        <label>
          论断蕴含校验（NLI）
          <select
            value={settings.nli_enabled || "true"}
            onChange={(event) => setSettings({ ...settings, nli_enabled: event.target.value })}
          >
            <option value="true">启用</option>
            <option value="false">关闭</option>
          </select>
        </label>
        <label>
          NLI 引擎
          <select
            value={settings.nli_engine || "none"}
            onChange={(event) => setSettings({ ...settings, nli_engine: event.target.value })}
          >
            <option value="none">仅规则校验（默认）</option>
            <option value="cross-encoder-nli">本地 cross-encoder NLI</option>
          </select>
        </label>
        <label>
          NLI 模型
          <input
            value={settings.nli_model || ""}
            onChange={(event) => setSettings({ ...settings, nli_model: event.target.value })}
            placeholder="如 cross-encoder/nli-deberta-v3-base"
          />
          <small className="field-help">需要安装 sentence-transformers；不可用时回退到规则校验。</small>
        </label>
        <label>
          上下文 token 预算
          <input
            type="number"
            min={2000}
            max={200000}
            value={settings.context_token_budget || "24000"}
            onChange={(event) => setSettings({ ...settings, context_token_budget: event.target.value })}
          />
          <small className="field-help">长对话会按此预算裁剪旧步骤，保留系统提示和原始问题。</small>
        </label>
        <button className="primary" type="submit">保存设置</button>
      </form>
      <div className="data-card">
        <h3>本地数据</h3>
        <p>数据目录：<code>{settings.data_dir || "默认用户目录\\HeartSeeker_Med"}</code></p>
        <p>包含原始 PDF、SQLite 数据库、解析缓存、LanceDB 向量索引和问答历史。</p>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
