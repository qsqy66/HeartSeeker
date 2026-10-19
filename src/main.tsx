import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as pdfjsLib from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import "./styles.css";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

type Paper = { id: number; title: string; file_name: string; page_count: number; status: string; error_message?: string; imported_at?: string; vectorized?: number; needs_ocr?: number };
type Page = { id: number; page_number: number; text: string };
type PaperDetails = { paper: Paper; pages: Page[] };
type SearchResult = { id: number; text: string; page_number: number; section?: string; heading?: string; score?: number | null; x0?: number; y0?: number; x1?: number; y1?: number; source?: "fts" | "vector" };
type ChatSource = { id: number; text: string; page_number: number; section?: string; heading?: string; x0?: number; y0?: number; x1?: number; y1?: number };
type ChatMessage = { id?: number; role: "user" | "assistant"; content: string; citations?: SearchResult[]; sources?: ChatSource[]; created_at?: string; error?: string };
type Settings = { provider: "deepseek" | "qwen" | ""; model: string; embedding_model: string; api_key_configured: boolean; data_dir: string };
type Highlight = { page: number; bbox: [number, number, number, number] | null };
type Structured = { fields: Record<string, { value: string; evidence: SearchResult[] }> };

const API = "http://127.0.0.1:8765/api";

async function readResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null) as { detail?: string } | null;
  if (!response.ok) throw new Error(payload?.detail || `请求失败（HTTP ${response.status}）`);
  return payload as T;
}

async function readError(response: Response): Promise<string> {
  const payload = await response.json().catch(() => null) as { detail?: string } | null;
  return payload?.detail || `请求失败（HTTP ${response.status}）`;
}

async function consumeStream(response: Response, onDelta: (text: string) => void): Promise<{ done?: { id?: number; content?: string; citations?: SearchResult[]; sources?: ChatSource[] }; error?: string }> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("当前浏览器不支持流式读取");
  const decoder = new TextDecoder();
  let buffer = "";
  const result: { done?: { id?: number; content?: string; citations?: SearchResult[]; sources?: ChatSource[] }; error?: string } = {};
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
        const obj = JSON.parse(trimmed) as { delta?: string; done?: { id?: number; content?: string; citations?: SearchResult[]; sources?: ChatSource[] }; error?: string };
        if (obj.delta) onDelta(obj.delta);
        else if (obj.done) result.done = obj.done;
        else if (obj.error) result.error = obj.error;
      } catch { /* 忽略未完成的半行 */ }
    }
  }
  return result;
}

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<PaperDetails | null>(null);
  const [tab, setTab] = useState<"library" | "reader" | "settings">("library");
  const [readerTab, setReaderTab] = useState<"read" | "insights" | "chat">("read");
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState<"vector" | "keyword">("vector");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [highlight, setHighlight] = useState<Highlight | null>(null);
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [structured, setStructured] = useState<Structured | null>(null);
  const [settings, setSettings] = useState<Settings>({ provider: "", model: "", embedding_model: "text-embedding-v3", api_key_configured: false, data_dir: "" });
  const [apiKey, setApiKey] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [generalChat, setGeneralChat] = useState<ChatMessage[]>([]);
  const [generalQuestion, setGeneralQuestion] = useState("");

  const loadPapers = async () => {
    try { setPapers(await readResponse<Paper[]>(await fetch(`${API}/papers`))); }
    catch (error) { setNotice(`无法连接本地服务：${error instanceof Error ? error.message : "未知错误"}`); }
  };
  const loadSettings = async () => {
    try { setSettings(await readResponse<Settings>(await fetch(`${API}/settings`))); } catch { /* older backend */ }
  };
  const loadGeneralChat = async () => {
    try { setGeneralChat(await readResponse<ChatMessage[]>(await fetch(`${API}/chat`))); } catch { /* older backend */ }
  };
  useEffect(() => { void loadPapers(); void loadSettings(); void loadGeneralChat(); }, []);

  const selectPaper = async (id: number) => {
    try {
      setSelected(await readResponse<PaperDetails>(await fetch(`${API}/papers/${id}`)));
      setTab("reader"); setReaderTab("read"); setResults([]); setChat([]); setHighlight(null);
      const response = await fetch(`${API}/papers/${id}/structured`);
      if (response.ok) setStructured(await response.json());
      const history = await fetch(`${API}/papers/${id}/chat`);
      if (history.ok) setChat(await history.json());
    } catch (error) { setNotice(error instanceof Error ? error.message : "读取论文失败"); }
  };
  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; if (!file) return;
    setBusy(true); const body = new FormData(); body.append("file", file);
    try {
      const payload = await readResponse<{ id: number }>(await fetch(`${API}/papers/import`, { method: "POST", body }));
      setNotice("论文已导入、解析并建立本地索引。"); await loadPapers(); await selectPaper(payload.id);
    } catch (error) { setNotice(`导入失败：${error instanceof Error ? error.message : "未知错误"}`); }
    finally { setBusy(false); event.target.value = ""; }
  };
  const search = async () => {
    if (!selected || !query.trim()) return setResults([]);
    const endpoint = searchMode === "vector" ? "retrieve" : "search";
    try { setResults(await readResponse<SearchResult[]>(await fetch(`${API}/papers/${selected.paper.id}/${endpoint}?q=${encodeURIComponent(query)}`))); }
    catch (error) { setNotice(error instanceof Error ? error.message : "检索失败"); }
  };
  const jumpToEvidence = (result: { page_number: number; x0?: number; y0?: number; x1?: number; y1?: number }) => {
    const hasBox = result.x0 != null && result.y0 != null && result.x1 != null && result.y1 != null;
    const bbox: [number, number, number, number] | null = hasBox ? [result.x0!, result.y0!, result.x1!, result.y1!] : null;
    setHighlight({ page: result.page_number, bbox });
    setReaderTab("read");
  };
  const streamPaper = async (content: string, retry: boolean) => {
    if (!selected) return;
    setChat((messages) => [...messages, { role: "assistant", content: "" }]);
    setBusy(true);
    try {
      const response = await fetch(`${API}/papers/${selected.paper.id}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: content, stream: true, retry }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const { done, error } = await consumeStream(response, (delta) => {
        setChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, content: msg.content + delta } : msg));
      });
      if (error) {
        setChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, content: `⚠ ${error}`, error } : msg));
      } else if (done) {
        setChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, id: done.id, citations: done.citations, sources: done.sources } : msg));
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : "未知错误";
      setChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, content: `⚠ ${reason}`, error: reason } : msg));
    } finally { setBusy(false); }
  };
  const ask = async (event: FormEvent) => {
    event.preventDefault(); if (!selected || !question.trim()) return;
    const content = question;
    setChat((messages) => [...messages, { role: "user", content }]); setQuestion("");
    await streamPaper(content, false);
  };
  const retryChat = async () => {
    if (!selected) return;
    const lastUser = chat.map((m) => m.role).lastIndexOf("user");
    if (lastUser < 0) return;
    const content = chat[lastUser].content;
    setChat((messages) => messages.slice(0, lastUser + 1));
    await streamPaper(content, true);
  };
  const streamGeneral = async (content: string, retry: boolean) => {
    setGeneralChat((messages) => [...messages, { role: "assistant", content: "" }]);
    setBusy(true);
    try {
      const response = await fetch(`${API}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: content, stream: true, retry }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const { error } = await consumeStream(response, (delta) => {
        setGeneralChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, content: msg.content + delta } : msg));
      });
      if (error) {
        setGeneralChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, content: `⚠ ${error}`, error } : msg));
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : "未知错误";
      setGeneralChat((messages) => messages.map((msg, i) => i === messages.length - 1 ? { ...msg, content: `⚠ ${reason}`, error: reason } : msg));
    } finally { setBusy(false); }
  };
  const askGeneral = async (event: FormEvent) => {
    event.preventDefault(); if (!generalQuestion.trim()) return;
    const content = generalQuestion;
    setGeneralChat((messages) => [...messages, { role: "user", content }]); setGeneralQuestion("");
    await streamGeneral(content, false);
  };
  const retryGeneral = async () => {
    const lastUser = generalChat.map((m) => m.role).lastIndexOf("user");
    if (lastUser < 0) return;
    const content = generalChat[lastUser].content;
    setGeneralChat((messages) => messages.slice(0, lastUser + 1));
    await streamGeneral(content, true);
  };
  const remove = async () => {
    if (!selected || !window.confirm("删除论文、解析文本、索引、问答和本地 PDF？")) return;
    try { await readResponse(await fetch(`${API}/papers/${selected.paper.id}`, { method: "DELETE" })); setSelected(null); setTab("library"); await loadPapers(); setNotice("论文及其本地关联数据已删除。"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "删除失败"); }
  };
  const saveSettings = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true);
    try {
      setSettings(await readResponse<Settings>(await fetch(`${API}/settings`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...settings, api_key: apiKey }),
      }))); setApiKey(""); setNotice("模型设置已保存。API Key 仅保存到本机安全存储，不写入日志或数据库。");
    } catch (error) { setNotice(`设置保存失败：${error instanceof Error ? error.message : "未知错误"}`); }
    finally { setBusy(false); }
  };
  const loadModels = async () => {
    if (!settings.provider) return setNotice("请先选择模型供应商。");
    try {
      const payload = await readResponse<{ models: string[] }>(await fetch(`${API}/models`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: settings.provider, api_key: apiKey || undefined }),
      }));
      setAvailableModels(payload.models);
      setNotice(`已读取 ${payload.models.length} 个可用模型。`);
    } catch (error) {
      setNotice(`读取模型列表失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  return <div className="app">
    <header className="topbar"><div className="brand"><span className="brand-mark">P</span><div><span className="eyebrow">PAPER READER</span><h1>可追溯的医学论文精读</h1></div></div><label className="import-button"><input type="file" accept=".pdf,application/pdf" onChange={importFile} />{busy ? "处理中…" : "导入 PDF"}</label></header>
    {notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice("")}>×</button></div>}
    <div className="layout">
      <nav className="nav"><button className={tab === "library" ? "nav-active" : ""} onClick={() => setTab("library")}>◫ <span>论文库</span></button>{selected && <button className={tab === "reader" ? "nav-active" : ""} onClick={() => setTab("reader")}>▣ <span>论文工作台</span></button>}<button className={tab === "settings" ? "nav-active" : ""} onClick={() => setTab("settings")}>⚙ <span>设置与隐私</span></button><div className="nav-foot">本地优先<br /><small>数据不会自动上传</small></div></nav>
      <main className="main">
        {tab === "library" && <><GeneralChat chat={generalChat} question={generalQuestion} setQuestion={setGeneralQuestion} ask={askGeneral} busy={busy} onRetry={retryGeneral} /><Library papers={papers} selected={selected} onSelect={(id) => void selectPaper(id)} onImport={importFile} /></>}
        {tab === "settings" && <SettingsView settings={settings} apiKey={apiKey} setApiKey={setApiKey} setSettings={setSettings} onSave={saveSettings} availableModels={availableModels} loadModels={() => void loadModels()} />}
        {tab === "reader" && selected && <Reader details={selected} readerTab={readerTab} setReaderTab={setReaderTab} query={query} setQuery={setQuery} searchMode={searchMode} setSearchMode={setSearchMode} results={results} search={() => void search()} highlight={highlight} onJump={jumpToEvidence} structured={structured} chat={chat} question={question} setQuestion={setQuestion} ask={ask} busy={busy} onRetry={retryChat} remove={() => void remove()} />}
      </main>
    </div>
  </div>;
}

function GeneralChat({ chat, question, setQuestion, ask, busy, onRetry }: { chat: ChatMessage[]; question: string; setQuestion: (value: string) => void; ask: (event: FormEvent) => void; busy: boolean; onRetry: () => void }) { return <section className="general-chat"><div className="chat-intro"><span className="eyebrow">MEDICAL LEARNING ASSISTANT</span><h3>AI 学习助手 · 通用医学对话</h3><p>不绑定论文的通用问答：解释概念、梳理机制、讨论病例。可连续追问，最近 10 轮完整保留，更早的自动压缩成摘要。</p></div><div className="messages general-messages">{chat.length ? chat.map((message, index) => <div className={`message ${message.role}`} key={message.id || index}><span>{message.role === "user" ? "你" : "AI 助手"}</span><p>{message.content}</p>{message.error && <button className="retry-button" onClick={onRetry}>↻ 重试</button>}</div>) : <div className="empty-state compact"><h3>开始提问</h3><p>例如：急性心衰和慢性心衰的治疗原则有什么区别？</p></div>}</div><form className="question-form" onSubmit={ask}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="问一个医学问题，可连续追问…" /><button disabled={busy}>发送</button></form></section>; }

function Library({ papers, selected, onSelect, onImport }: { papers: Paper[]; selected: PaperDetails | null; onSelect: (id: number) => void; onImport: (event: ChangeEvent<HTMLInputElement>) => void }) {
  return <section className="library"><div className="hero"><div><span className="eyebrow">LOCAL RESEARCH DESK</span><h2>你的论文库</h2><p>导入英文医学论文，在本机完成解析、检索与可核验阅读。</p></div><label className="primary"><input type="file" accept=".pdf,application/pdf" onChange={onImport} />＋ 导入论文</label></div><div className="privacy-card"><b>隐私边界</b><span>PDF、解析文本、切片、向量和问答历史默认只保存在本机。只有主动使用模型功能时，才会发送问题和必要原文片段。</span></div><div className="section-heading"><h3>全部论文</h3><span>{papers.length} 篇</span></div>{papers.length ? <div className="paper-grid">{papers.map((paper) => <button className={`paper-card ${selected?.paper.id === paper.id ? "selected" : ""}`} key={paper.id} onClick={() => onSelect(paper.id)}><div className="paper-icon">PDF</div><div className="paper-info"><h3>{paper.title}</h3><p>{paper.file_name}</p><small>{paper.page_count} 页 · {paper.status === "ready" ? (paper.needs_ocr ? "需 OCR" : "已完成") : paper.status}</small></div><span className="arrow">›</span></button>)}</div> : <div className="empty-state"><div className="empty-icon">＋</div><h3>还没有论文</h3><p>导入一篇英文医学 PDF，开始建立你的本地证据库。</p></div>}</section>;
}

function Reader({ details, readerTab, setReaderTab, query, setQuery, searchMode, setSearchMode, results, search, highlight, onJump, structured, chat, question, setQuestion, ask, busy, onRetry, remove }: { details: PaperDetails; readerTab: "read" | "insights" | "chat"; setReaderTab: (tab: "read" | "insights" | "chat") => void; query: string; setQuery: (value: string) => void; searchMode: "vector" | "keyword"; setSearchMode: (mode: "vector" | "keyword") => void; results: SearchResult[]; search: () => void; highlight: Highlight | null; onJump: (result: { page_number: number; x0?: number; y0?: number; x1?: number; y1?: number }) => void; structured: Structured | null; chat: ChatMessage[]; question: string; setQuestion: (value: string) => void; ask: (event: FormEvent) => void; busy: boolean; onRetry: () => void; remove: () => void }) {
  return <section className="reader"><div className="reader-head"><div><span className="eyebrow">论文工作台</span><h2>{details.paper.title}</h2><p>{details.paper.file_name} · {details.paper.page_count} 页</p></div><div className="head-actions"><a href={`${API}/papers/${details.paper.id}/file`} target="_blank" rel="noreferrer">打开 PDF</a><button className="danger" onClick={remove}>删除</button></div></div><div className="reader-tabs"><button className={readerTab === "read" ? "active" : ""} onClick={() => setReaderTab("read")}>原文阅读</button><button className={readerTab === "insights" ? "active" : ""} onClick={() => setReaderTab("insights")}>结构化精读</button><button className={readerTab === "chat" ? "active" : ""} onClick={() => setReaderTab("chat")}>论文内问答</button></div>{readerTab === "read" && <div className="reader-grid"><div className="pdf-column"><div className="searchbar"><select className="search-mode" value={searchMode} onChange={(event) => setSearchMode(event.target.value as "vector" | "keyword")}><option value="vector">语义检索</option><option value="keyword">关键词检索</option></select><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && search()} placeholder="搜索原文，如 primary outcome" /><button onClick={search}>检索</button></div><PdfViewer fileUrl={`${API}/papers/${details.paper.id}/file`} pageCount={details.paper.page_count} highlight={highlight} /></div><Evidence results={results} onJump={onJump} /></div>}{readerTab === "insights" && <Insights structured={structured} onJump={onJump} />}{readerTab === "chat" && <Chat chat={chat} question={question} setQuestion={setQuestion} ask={ask} onJump={onJump} busy={busy} onRetry={onRetry} />}</section>;
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
  return <div className="pdf-viewer">{pageNumbers.map((n) => <PdfPage key={n} pdf={pdf} pageNumber={n} highlight={highlight && highlight.page === n ? highlight : null} />)}</div>;
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
      if (entries.some((entry) => entry.isIntersecting)) { setRendered(true); observer.disconnect(); }
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
      } catch (error) { console.error("渲染 PDF 页面失败", error); }
    })();
    return () => { cancelled = true; };
  }, [rendered, pdf, pageNumber]);

  const highlightStyle = highlight && highlight.bbox && baseViewport ? {
    left: `${(highlight.bbox[0] / baseViewport.width) * 100}%`,
    top: `${(highlight.bbox[1] / baseViewport.height) * 100}%`,
    width: `${((highlight.bbox[2] - highlight.bbox[0]) / baseViewport.width) * 100}%`,
    height: `${((highlight.bbox[3] - highlight.bbox[1]) / baseViewport.height) * 100}%`,
  } : undefined;

  return <div className="pdf-page" id={`page-${pageNumber}`} ref={wrapRef}><div className="pdf-page-label">PDF 第 {pageNumber} 页</div><div className="pdf-canvas-wrap">{rendered ? <canvas ref={canvasRef} style={{ width: "100%", height: "auto", display: "block" }} /> : <div className="pdf-page-placeholder">滚动加载…</div>}{highlightStyle && <div className="pdf-highlight" style={highlightStyle} />}</div></div>;
}

function Evidence({ results, onJump }: { results: SearchResult[]; onJump: (result: SearchResult) => void }) { return <aside className="evidence-panel"><div className="panel-title"><h3>引用证据</h3><span>{results.length} 条</span></div>{results.length ? results.map((result) => <button className="evidence-item" key={result.id} onClick={() => onJump(result)}><span>PDF 第 {result.page_number} 页{result.section && ` · ${result.section}`}{result.source === "vector" && " · 语义"}{result.score != null && ` · 距离 ${result.score}`}</span><p>{result.text}</p><small>切片 #{result.id} · 点击高亮原文</small></button>) : <p className="empty">检索结果会显示在这里，点击可在 PDF 上高亮对应位置。</p>}</aside>; }
function Insights({ structured, onJump }: { structured: Structured | null; onJump: (result: SearchResult) => void }) { const fields = structured?.fields || {}; return <div className="insights"><div className="insight-intro"><span className="eyebrow">EVIDENCE-GROUNDED SUMMARY</span><h3>结构化论文精读</h3><p>没有明确原文证据的字段会标记为“论文未明确报告”，不会自动猜测。</p></div><div className="insight-grid">{["一句话总结", "研究设计", "Population", "Intervention / Exposure", "Comparator", "Outcomes", "样本量", "统计方法", "关键结果", "局限性"].map((name) => <article className="insight-card" key={name}><span>{name}</span><strong>{fields[name]?.value || "论文未明确报告"}</strong>{fields[name]?.evidence?.map((evidence) => <button className="source-chip" key={evidence.id} onClick={() => onJump(evidence)}>PDF 第 {evidence.page_number} 页 · {evidence.text.slice(0, 120)}</button>)}</article>)}</div></div>; }
function Chat({ chat, question, setQuestion, ask, onJump, busy, onRetry }: { chat: ChatMessage[]; question: string; setQuestion: (value: string) => void; ask: (event: FormEvent) => void; onJump: (result: { page_number: number; x0?: number; y0?: number; x1?: number; y1?: number }) => void; busy: boolean; onRetry: () => void }) { return <div className="chat"><div className="chat-intro"><span className="eyebrow">ONE-PAPER RAG</span><h3>只基于当前论文提问</h3><p>回答必须附带本地证据；论文没有报告的信息会明确说明证据不足。</p></div><div className="messages">{chat.length ? chat.map((message, index) => <div className={`message ${message.role}`} key={message.id || index}><span>{message.role === "user" ? "你" : "Paper Reader"}</span><p>{message.content}</p>{message.citations?.map((citation) => <small key={citation.id}>引用：PDF 第 {citation.page_number} 页</small>)}{message.sources?.map((source) => <button className="source-chip" key={source.id} onClick={() => onJump(source)}>证据 · PDF 第 {source.page_number} 页{source.section && ` · ${source.section}`}：{source.text.slice(0, 90)}</button>)}{message.error && <button className="retry-button" onClick={onRetry}>↻ 重试</button>}</div>) : <div className="empty-state compact"><h3>还没有问题</h3><p>例如：这项研究的主要终点是什么？</p></div>}</div><form className="question-form" onSubmit={ask}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="输入关于当前论文的问题…" /><button disabled={busy}>发送</button></form></div>; }
function SettingsView({ settings, apiKey, setApiKey, setSettings, onSave, availableModels, loadModels }: { settings: Settings; apiKey: string; setApiKey: (value: string) => void; setSettings: (value: Settings) => void; onSave: (event: FormEvent) => void; availableModels: string[]; loadModels: () => void }) { return <section className="settings"><span className="eyebrow">LOCAL CONFIGURATION</span><h2>设置与隐私</h2><p className="subtitle">模型服务只在你主动发起精读或问答时使用。</p><form onSubmit={onSave} className="settings-form"><label>模型供应商<select value={settings.provider} onChange={(event) => setSettings({ ...settings, provider: event.target.value as Settings["provider"], model: "" })}><option value="">未配置</option><option value="deepseek">DeepSeek</option><option value="qwen">Qwen / 通义千问</option></select></label><label>API Key<input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={settings.api_key_configured ? "已配置（输入新 Key 可替换）" : "输入后可读取模型列表"} /></label><div className="model-row"><select value={availableModels.includes(settings.model) ? settings.model : ""} onChange={(event) => setSettings({ ...settings, model: event.target.value })}><option value="">选择已读取的模型</option>{availableModels.map((model) => <option key={model} value={model}>{model}</option>)}</select><button type="button" onClick={loadModels}>读取可用模型</button></div><label>模型名称<input value={settings.model} onChange={(event) => setSettings({ ...settings, model: event.target.value })} placeholder="也可手动输入模型 ID" /><small className="field-help">读取列表后可直接选择；保存后问答请求会使用这里的名称。</small></label><label>嵌入模型<input value={settings.embedding_model} onChange={(event) => setSettings({ ...settings, embedding_model: event.target.value })} placeholder="text-embedding-v3" /><small className="field-help">用于语义检索与问答召回的向量模型（千问/DashScope，向量仍存本机）。</small></label><div className="security-note">读取模型列表和远端问答都会向所选供应商发送请求。API Key 不会发送给开发者服务器，也不会写入论文数据库。</div><button className="primary" type="submit">保存设置</button></form><div className="data-card"><h3>本地数据</h3><p>数据目录：<code>{settings.data_dir || "默认用户目录\\PaperReader"}</code></p><p>包含原始 PDF、SQLite 数据库、解析缓存、LanceDB 向量索引和问答历史。</p></div></section>; }

createRoot(document.getElementById("root")!).render(<App />);
