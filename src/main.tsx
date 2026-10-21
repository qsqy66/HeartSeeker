import {
  ChangeEvent,
  Dispatch,
  FormEvent,
  SetStateAction,
  useEffect,
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
type Settings = {
  provider: "deepseek" | "qwen" | "";
  model: string;
  embedding_model: string;
  api_key_configured: boolean;
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
        {message.verified != null && (
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
  return (
    <div className={`message ${message.role}`}>
      <span>{message.role === "user" ? "你" : "Paper Reader"}</span>
      {message.role === "assistant" && message.pending && !message.content ? (
        <p className="typing">正在检索证据并核验引用…</p>
      ) : (
        <Markdown content={message.content} onJump={onJump} />
      )}
      <AgentTrace message={message} />
      {message.citations && message.citations.length > 0 && (
        <CitationChips citations={message.citations} onJump={onJump} />
      )}
      {message.sources && message.sources.length > 0 && (
        <div className="source-list">
          {message.sources.map((source) => (
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
    </div>
  );
}

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<PaperDetails | null>(null);
  const [tab, setTab] = useState<"library" | "reader" | "settings">("library");
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
    model: "",
    embedding_model: "text-embedding-v3",
    api_key_configured: false,
    data_dir: "",
  });
  const [apiKey, setApiKey] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const [generalChat, setGeneralChat] = useState<ChatMessage[]>([]);
  const [generalQuestion, setGeneralQuestion] = useState("");
  const [generalMode, setGeneralMode] = useState<GeneralMode>("library");
  const [libraryQuery, setLibraryQuery] = useState("");
  const [libraryResults, setLibraryResults] = useState<SearchResult[]>([]);
  const [librarySearching, setLibrarySearching] = useState(false);

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
  const loadGeneralChat = async () => {
    try {
      setGeneralChat(await readResponse<ChatMessage[]>(await fetch(`${API}/chat`)));
    } catch {
      /* older backend */
    }
  };

  useEffect(() => {
    void loadPapers();
    void loadSettings();
    void loadGeneralChat();
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
  ) => {
    setMessages((messages) => [...messages, { role: "assistant", content: "", pending: true }]);
    setBusy(true);
    try {
      const response = await fetch(`${API}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, mode, paper_id: paperId, stream: true, retry }),
      });
      if (!response.ok) throw new Error(await readError(response));
      let finished = false;
      await consumeAgentStream(response, (event) => {
        if (event.type === "step") {
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
      const reason = reasonOf(error);
      patchLast(setMessages, (message) => ({
        ...message,
        content: message.content || `⚠ ${reason}`,
        error: reason,
        pending: false,
      }));
    } finally {
      setBusy(false);
    }
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
    if (!generalQuestion.trim()) return;
    const content = generalQuestion;
    const mode: GeneralMode = papers.length > 0 ? generalMode : "general";
    setGeneralChat((messages) => [...messages, { role: "user", content }]);
    setGeneralQuestion("");
    await runAgent(mode, content, false, undefined, setGeneralChat);
  };

  const retryGeneral = async () => {
    const lastUser = generalChat.map((message) => message.role).lastIndexOf("user");
    if (lastUser < 0) return;
    const content = generalChat[lastUser].content;
    const mode: GeneralMode = papers.length > 0 ? generalMode : "general";
    setGeneralChat((messages) => messages.slice(0, lastUser + 1));
    await runAgent(mode, content, true, undefined, setGeneralChat);
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
        body: JSON.stringify({ ...settings, api_key: apiKey }),
      })));
      setApiKey("");
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
        body: JSON.stringify({ provider: settings.provider, api_key: apiKey || undefined }),
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
          <span className="brand-mark">P</span>
          <div>
            <span className="eyebrow">PAPER READER</span>
            <h1>可追溯的医学论文精读</h1>
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
          <button className={tab === "library" ? "nav-active" : ""} onClick={() => setTab("library")}>
            ◫ <span>论文库</span>
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
          {tab === "library" && (
            <>
              <GeneralChat
                chat={generalChat}
                question={generalQuestion}
                setQuestion={setGeneralQuestion}
                ask={askGeneral}
                busy={busy}
                onRetry={() => void retryGeneral()}
                mode={generalMode}
                setMode={setGeneralMode}
                hasPapers={papers.length > 0}
                onJump={jumpTo}
              />
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
            </>
          )}
          {tab === "settings" && (
            <SettingsView
              settings={settings}
              apiKey={apiKey}
              setApiKey={setApiKey}
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

function GeneralChat({
  chat,
  question,
  setQuestion,
  ask,
  busy,
  onRetry,
  mode,
  setMode,
  hasPapers,
  onJump,
}: {
  chat: ChatMessage[];
  question: string;
  setQuestion: (value: string) => void;
  ask: (event: FormEvent) => void;
  busy: boolean;
  onRetry: () => void;
  mode: GeneralMode;
  setMode: (mode: GeneralMode) => void;
  hasPapers: boolean;
  onJump: JumpHandler;
}) {
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
      <div className="messages general-messages">
        {chat.length ? chat.map((message, index) => (
          <MessageView key={message.id || index} message={message} onJump={onJump} onRetry={onRetry} />
        )) : (
          <div className="empty-state compact">
            <h3>开始提问</h3>
            <p>例如：急性心衰和慢性心衰的治疗原则有什么区别？</p>
          </div>
        )}
      </div>
      <form className="question-form" onSubmit={ask}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="问一个医学问题，可连续追问…"
        />
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
  onRetry,
}: {
  chat: ChatMessage[];
  question: string;
  setQuestion: (value: string) => void;
  ask: (event: FormEvent) => void;
  onJump: JumpHandler;
  busy: boolean;
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
        <button disabled={busy}>发送</button>
      </form>
    </div>
  );
}

function SettingsView({
  settings,
  apiKey,
  setApiKey,
  setSettings,
  onSave,
  availableModels,
  loadModels,
}: {
  settings: Settings;
  apiKey: string;
  setApiKey: (value: string) => void;
  setSettings: (value: Settings) => void;
  onSave: (event: FormEvent) => void;
  availableModels: string[];
  loadModels: () => void;
}) {
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
            onChange={(event) => setSettings({
              ...settings,
              provider: event.target.value as Settings["provider"],
              model: "",
            })}
          >
            <option value="">未配置</option>
            <option value="deepseek">DeepSeek</option>
            <option value="qwen">Qwen / 通义千问</option>
          </select>
        </label>
        <label>
          API Key
          <input
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder={settings.api_key_configured ? "已配置（输入新 Key 可替换）" : "输入后可读取模型列表"}
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
        <label>
          嵌入模型
          <input
            value={settings.embedding_model}
            onChange={(event) => setSettings({ ...settings, embedding_model: event.target.value })}
            placeholder="text-embedding-v3"
          />
          <small className="field-help">用于语义检索与问答召回的向量模型（千问/DashScope，向量仍存本机）。</small>
        </label>
        <div className="security-note">
          读取模型列表和远端问答都会向所选供应商发送请求。API Key 不会发送给开发者服务器，也不会写入论文数据库。
        </div>
        <button className="primary" type="submit">保存设置</button>
      </form>
      <div className="data-card">
        <h3>本地数据</h3>
        <p>数据目录：<code>{settings.data_dir || "默认用户目录\\PaperReader"}</code></p>
        <p>包含原始 PDF、SQLite 数据库、解析缓存、LanceDB 向量索引和问答历史。</p>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
