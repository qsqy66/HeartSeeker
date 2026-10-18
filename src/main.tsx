import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Paper = { id: number; title: string; file_name: string; page_count: number; status: string; error_message?: string; imported_at?: string };
type Page = { id: number; page_number: number; text: string };
type PaperDetails = { paper: Paper; pages: Page[] };
type SearchResult = { id: number; text: string; page_number: number; section?: string };
type Structured = { fields: Record<string, { value: string; evidence: SearchResult[] }> };
type ChatMessage = { id?: number; role: "user" | "assistant"; content: string; citations?: SearchResult[]; created_at?: string };
type Settings = { provider: "deepseek" | "qwen" | ""; model: string; api_key_configured: boolean; data_dir: string };

const API = "http://127.0.0.1:8765/api";

async function readResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null) as { detail?: string } | null;
  if (!response.ok) throw new Error(payload?.detail || `请求失败（HTTP ${response.status}）`);
  return payload as T;
}

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<PaperDetails | null>(null);
  const [tab, setTab] = useState<"library" | "reader" | "settings">("library");
  const [readerTab, setReaderTab] = useState<"read" | "insights" | "chat">("read");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [structured, setStructured] = useState<Structured | null>(null);
  const [settings, setSettings] = useState<Settings>({ provider: "", model: "", api_key_configured: false, data_dir: "" });
  const [apiKey, setApiKey] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const loadPapers = async () => {
    try { setPapers(await readResponse<Paper[]>(await fetch(`${API}/papers`))); }
    catch (error) { setNotice(`无法连接本地服务：${error instanceof Error ? error.message : "未知错误"}`); }
  };
  const loadSettings = async () => {
    try { setSettings(await readResponse<Settings>(await fetch(`${API}/settings`))); } catch { /* older backend */ }
  };
  useEffect(() => { void loadPapers(); void loadSettings(); }, []);

  const selectPaper = async (id: number) => {
    try {
      setSelected(await readResponse<PaperDetails>(await fetch(`${API}/papers/${id}`)));
      setTab("reader"); setReaderTab("read"); setResults([]); setChat([]);
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
    try { setResults(await readResponse<SearchResult[]>(await fetch(`${API}/papers/${selected.paper.id}/search?q=${encodeURIComponent(query)}`))); }
    catch (error) { setNotice(error instanceof Error ? error.message : "检索失败"); }
  };
  const ask = async (event: FormEvent) => {
    event.preventDefault(); if (!selected || !question.trim()) return;
    const userMessage: ChatMessage = { role: "user", content: question };
    setChat((messages) => [...messages, userMessage]); setQuestion(""); setBusy(true);
    try {
      const answer = await readResponse<ChatMessage>(await fetch(`${API}/papers/${selected.paper.id}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: userMessage.content }),
      }));
      setChat((messages) => [...messages, answer]);
    } catch (error) { setNotice(`问答失败：${error instanceof Error ? error.message : "未知错误"}`); }
    finally { setBusy(false); }
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
        {tab === "library" && <Library papers={papers} selected={selected} onSelect={(id) => void selectPaper(id)} onImport={importFile} />}
        {tab === "settings" && <SettingsView settings={settings} apiKey={apiKey} setApiKey={setApiKey} setSettings={setSettings} onSave={saveSettings} availableModels={availableModels} loadModels={() => void loadModels()} />}
        {tab === "reader" && selected && <Reader details={selected} readerTab={readerTab} setReaderTab={setReaderTab} query={query} setQuery={setQuery} results={results} search={() => void search()} structured={structured} chat={chat} question={question} setQuestion={setQuestion} ask={ask} remove={() => void remove()} />}
      </main>
    </div>
  </div>;
}

function Library({ papers, selected, onSelect, onImport }: { papers: Paper[]; selected: PaperDetails | null; onSelect: (id: number) => void; onImport: (event: ChangeEvent<HTMLInputElement>) => void }) {
  return <section className="library"><div className="hero"><div><span className="eyebrow">LOCAL RESEARCH DESK</span><h2>你的论文库</h2><p>导入英文医学论文，在本机完成解析、检索与可核验阅读。</p></div><label className="primary"><input type="file" accept=".pdf,application/pdf" onChange={onImport} />＋ 导入论文</label></div><div className="privacy-card"><b>隐私边界</b><span>PDF、解析文本、切片、向量和问答历史默认只保存在本机。只有主动使用模型功能时，才会发送问题和必要原文片段。</span></div><div className="section-heading"><h3>全部论文</h3><span>{papers.length} 篇</span></div>{papers.length ? <div className="paper-grid">{papers.map((paper) => <button className={`paper-card ${selected?.paper.id === paper.id ? "selected" : ""}`} key={paper.id} onClick={() => onSelect(paper.id)}><div className="paper-icon">PDF</div><div className="paper-info"><h3>{paper.title}</h3><p>{paper.file_name}</p><small>{paper.page_count} 页 · {paper.status === "ready" ? "已完成" : paper.status}</small></div><span className="arrow">›</span></button>)}</div> : <div className="empty-state"><div className="empty-icon">＋</div><h3>还没有论文</h3><p>导入一篇英文医学 PDF，开始建立你的本地证据库。</p></div>}</section>;
}

function Reader({ details, readerTab, setReaderTab, query, setQuery, results, search, structured, chat, question, setQuestion, ask, remove }: { details: PaperDetails; readerTab: "read" | "insights" | "chat"; setReaderTab: (tab: "read" | "insights" | "chat") => void; query: string; setQuery: (value: string) => void; results: SearchResult[]; search: () => void; structured: Structured | null; chat: ChatMessage[]; question: string; setQuestion: (value: string) => void; ask: (event: FormEvent) => void; remove: () => void }) {
  return <section className="reader"><div className="reader-head"><div><span className="eyebrow">论文工作台</span><h2>{details.paper.title}</h2><p>{details.paper.file_name} · {details.paper.page_count} 页</p></div><div className="head-actions"><a href={`${API}/papers/${details.paper.id}/file`} target="_blank" rel="noreferrer">打开 PDF</a><button className="danger" onClick={remove}>删除</button></div></div><div className="reader-tabs"><button className={readerTab === "read" ? "active" : ""} onClick={() => setReaderTab("read")}>原文阅读</button><button className={readerTab === "insights" ? "active" : ""} onClick={() => setReaderTab("insights")}>结构化精读</button><button className={readerTab === "chat" ? "active" : ""} onClick={() => setReaderTab("chat")}>论文内问答</button></div>{readerTab === "read" && <div className="reader-grid"><div className="pdf-column"><div className="searchbar"><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && search()} placeholder="搜索原文关键词，如 primary outcome" /><button onClick={search}>检索</button></div>{details.pages.map((page) => <article className="page" id={`page-${page.page_number}`} key={page.id}><div className="page-label">PDF 第 {page.page_number} 页</div><p>{page.text || "本页未解析出可复制文本，可能需要 OCR。"}</p></article>)}</div><Evidence results={results} /></div>}{readerTab === "insights" && <Insights structured={structured} />}{readerTab === "chat" && <Chat chat={chat} question={question} setQuestion={setQuestion} ask={ask} />}</section>;
}

function Evidence({ results }: { results: SearchResult[] }) { return <aside className="evidence-panel"><div className="panel-title"><h3>引用证据</h3><span>{results.length} 条</span></div>{results.length ? results.map((result) => <button className="evidence-item" key={result.id} onClick={() => document.getElementById(`page-${result.page_number}`)?.scrollIntoView({ behavior: "smooth" })}><span>PDF 第 {result.page_number} 页 {result.section && `· ${result.section}`}</span><p>{result.text}</p><small>切片 #{result.id} · 点击跳转原文</small></button>) : <p className="empty">搜索结果会显示在这里，并可跳转到对应原文页。</p>}</aside>; }
function Insights({ structured }: { structured: Structured | null }) { const fields = structured?.fields || {}; return <div className="insights"><div className="insight-intro"><span className="eyebrow">EVIDENCE-GROUNDED SUMMARY</span><h3>结构化论文精读</h3><p>没有明确原文证据的字段会标记为“论文未明确报告”，不会自动猜测。</p></div><div className="insight-grid">{["一句话总结", "研究设计", "Population", "Intervention / Exposure", "Comparator", "Outcomes", "样本量", "统计方法", "关键结果", "局限性"].map((name) => <article className="insight-card" key={name}><span>{name}</span><strong>{fields[name]?.value || "论文未明确报告"}</strong>{fields[name]?.evidence?.map((evidence) => <small key={evidence.id}>PDF 第 {evidence.page_number} 页 · {evidence.text.slice(0, 120)}</small>)}</article>)}</div></div>; }
function Chat({ chat, question, setQuestion, ask }: { chat: ChatMessage[]; question: string; setQuestion: (value: string) => void; ask: (event: FormEvent) => void }) { return <div className="chat"><div className="chat-intro"><span className="eyebrow">ONE-PAPER RAG</span><h3>只基于当前论文提问</h3><p>回答必须附带本地证据；论文没有报告的信息会明确说明证据不足。</p></div><div className="messages">{chat.length ? chat.map((message, index) => <div className={`message ${message.role}`} key={message.id || index}><span>{message.role === "user" ? "你" : "Paper Reader"}</span><p>{message.content}</p>{message.citations?.map((citation) => <small key={citation.id}>引用：PDF 第 {citation.page_number} 页 · {citation.text.slice(0, 120)}</small>)}</div>) : <div className="empty-state compact"><h3>还没有问题</h3><p>例如：这项研究的主要终点是什么？</p></div>}</div><form className="question-form" onSubmit={ask}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="输入关于当前论文的问题…" /><button>发送</button></form></div>; }
function SettingsView({ settings, apiKey, setApiKey, setSettings, onSave, availableModels, loadModels }: { settings: Settings; apiKey: string; setApiKey: (value: string) => void; setSettings: (value: Settings) => void; onSave: (event: FormEvent) => void; availableModels: string[]; loadModels: () => void }) { return <section className="settings"><span className="eyebrow">LOCAL CONFIGURATION</span><h2>设置与隐私</h2><p className="subtitle">模型服务只在你主动发起精读或问答时使用。</p><form onSubmit={onSave} className="settings-form"><label>模型供应商<select value={settings.provider} onChange={(event) => setSettings({ ...settings, provider: event.target.value as Settings["provider"], model: "" })}><option value="">未配置</option><option value="deepseek">DeepSeek</option><option value="qwen">Qwen / 通义千问</option></select></label><label>API Key<input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={settings.api_key_configured ? "已配置（输入新 Key 可替换）" : "输入后可读取模型列表"} /></label><div className="model-row"><select value={availableModels.includes(settings.model) ? settings.model : ""} onChange={(event) => setSettings({ ...settings, model: event.target.value })}><option value="">选择已读取的模型</option>{availableModels.map((model) => <option key={model} value={model}>{model}</option>)}</select><button type="button" onClick={loadModels}>读取可用模型</button></div><label>模型名称<input value={settings.model} onChange={(event) => setSettings({ ...settings, model: event.target.value })} placeholder="也可手动输入模型 ID" /><small className="field-help">读取列表后可直接选择；保存后问答请求会使用这里的名称。</small></label><div className="security-note">读取模型列表和远端问答都会向所选供应商发送请求。API Key 不会发送给开发者服务器，也不会写入论文数据库。</div><button className="primary" type="submit">保存设置</button></form><div className="data-card"><h3>本地数据</h3><p>数据目录：<code>{settings.data_dir || "默认用户目录\\PaperReader"}</code></p><p>包含原始 PDF、SQLite 数据库、解析缓存、索引和问答历史。</p></div></section>; }

createRoot(document.getElementById("root")!).render(<App />);
