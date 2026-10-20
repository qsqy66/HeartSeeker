"""A small, auditable medical research agent.

The model is remote (DeepSeek/Qwen), but every step runs locally: the agent
chooses a tool, the tool queries the local SQLite/LanceDB indexes, and the
observation is appended to the transcript.  There is no hidden framework and
no client-side tool execution.

Protocol
--------
The model must answer with one JSON object per turn::

    {"thought": "...", "tool": "search_library", "args": {...}}
    {"thought": "...", "tool": "finish", "args": {"answer": "...", "citations": [...]}}

Using a JSON protocol instead of native function calling keeps the loop
portable across DeepSeek and Qwen and makes failures easy to inspect.
"""
from __future__ import annotations

import ast
import json
import operator
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import providers, retrieval
from .citations import PAGE_RE, citation_footer, validate_citations
from .safety import MEDICAL_SYSTEM_RULES, apply_safety, assess_question, wrap_untrusted

MAX_STEPS = 7
MAX_OBSERVATION_CHARS = 9000


class AgentError(RuntimeError):
    """Raised when an agent run cannot produce any usable result."""


@dataclass
class AgentConfig:
    provider: str
    key: str
    model: str
    mode: str = "paper"  # "paper" | "library" | "general"
    paper_id: Optional[int] = None
    max_steps: int = MAX_STEPS
    temperature: float = 0.1
    history: list[dict[str, Any]] = field(default_factory=list)


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "list_papers": {
        "description": "列出本地论文库中的论文（id、标题、页数）。",
        "args": {},
    },
    "search_library": {
        "description": "在本地论文库中做混合检索（语义+关键词），返回带页码的证据片段。",
        "args": {"query": "检索词或问题", "paper_id": "可选，限定某篇论文", "top_k": "可选，默认 8"},
    },
    "read_page": {
        "description": "读取某篇论文的某一页全文，用于核对上下文或引用。",
        "args": {"paper_id": "论文 id", "page": "页码，从 1 开始"},
    },
    "read_section": {
        "description": "按章节标题读取论文片段（如 Methods、Results、Limitations）。",
        "args": {"paper_id": "论文 id", "heading": "章节标题或关键词"},
    },
    "compare_papers": {
        "description": "针对多个方面分别检索多篇论文，用于跨研究对比。",
        "args": {"paper_ids": "论文 id 列表", "aspects": "对比维度列表，如 study design、sample size"},
    },
    "extract_paper": {
        "description": "用 LLM 抽取论文的 PICO、样本量、统计方法、关键结果、局限性等结构化字段。",
        "args": {"paper_id": "论文 id"},
    },
    "calculate": {
        "description": "安全计算器，用于 ARR/RRR/NNT 等换算。只支持数字、括号、+ - * / ** 和 abs/round/min/max。",
        "args": {"expression": "例如 1 / abs(0.20 - 0.10)"},
    },
    "finish": {
        "description": "提交最终回答。citations 必须使用工具返回值中的原文，逐字复制 quote。",
        "args": {
            "answer": "中文回答，可含 [Page N] 标记",
            "citations": "列表：[{paper_id, page, quote, claim}]",
        },
    },
}


def _system_prompt(config: AgentConfig) -> str:
    tools = json.dumps(TOOL_SPECS, ensure_ascii=False, indent=2)
    mode_rule = {
        "paper": "当前任务只针对一篇论文。引用必须来自当前论文；如确需其他论文，先说明原因。",
        "library": "当前任务是全库问答。涉及具体研究、数字或结论时必须检索并引用论文；通用医学概念可以解释，但不得伪造研究证据。",
        "general": "当前任务是通用医学学习问答。可以解释通用知识；一旦引用具体论文或研究数据，必须使用本地检索证据。",
    }.get(config.mode, "")
    return f"""你是面向医学生的循证医学研究助手。你可以调用本地工具检索论文库、精读页面、比较论文、抽取结构化字段和做计算。

【工作方式】
1. 每次只输出一个 JSON 对象，不要输出 Markdown 代码块或额外解释。
2. 需要信息时调用工具；不要凭记忆编造论文内容、样本量、统计量或页码。
3. 已获得足够证据后调用 finish。最多 {config.max_steps} 步。
4. 引用格式必须为 {{"paper_id": 数字, "page": 数字, "quote": "原文逐字摘录", "claim": "该引用支持的论断"}}。
5. quote 必须逐字复制自工具 Observation 中出现的论文原文，不得改写、概括或拼接；找不到原文时不要引用。
6. 如果证据不足，明确写“现有证据不足”，不要猜测。
7. 涉及多篇论文时，先 list_papers / search_library，再比较；不要假设论文内容。

【任务约束】
{mode_rule}
{MEDICAL_SYSTEM_RULES}

【可用工具】
{tools}

【输出格式】
调用工具：
{{"thought": "下一步理由", "tool": "工具名", "args": {{...}}}}
结束：
{{"thought": "为什么证据已足够", "tool": "finish", "args": {{"answer": "...", "citations": [...]}}}}
"""


def _initial_user_prompt(config: AgentConfig, question: str, papers: list[dict[str, Any]],
                         initial_rows: list[dict[str, Any]]) -> str:
    paper_lines = "\n".join(
        f"- id={paper['id']}：《{paper['title']}》 ({paper.get('page_count', '?')} 页)"
        for paper in papers[:40]
    ) or "（论文库为空）"
    evidence = retrieval.build_context(initial_rows, max_chars=16000, per_row_chars=1200)
    parts = ["【用户问题】\n" + question]
    if config.history:
        history_lines = [
            f"{'用户' if message.get('role') == 'user' else '助手'}: "
            f"{_clip(message.get('content'), 900)}"
            for message in config.history[-20:]
        ]
        parts.append("【最近对话】\n" + "\n".join(history_lines))
    parts.append("【本地论文库】\n" + paper_lines)
    if config.mode == "paper" and config.paper_id is not None:
        current = next((paper for paper in papers if int(paper["id"]) == int(config.paper_id)), None)
        if current:
            parts.append(f"【当前论文】\nid={current['id']}：《{current['title']}》 ({current.get('page_count', '?')} 页)")
    if evidence:
        parts.append("【初始检索证据】\n" + wrap_untrusted(evidence) +
                     "\n这些证据已加入引用核对池，可直接用于回答；如不足请继续调用工具。")
    return "\n\n".join(parts)


def _safe_calculate(expression: str) -> float:
    """Evaluate a narrowly scoped arithmetic expression without ``eval``."""
    if not expression or len(expression) > 200:
        raise ValueError("表达式为空或过长")
    binary = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    }
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    functions = {"abs": abs, "round": round, "min": min, "max": max}

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            return float(functions[node.func.id](*[evaluate(arg) for arg in node.args]))
        raise ValueError("不支持的表达式")

    return evaluate(ast.parse(expression, mode="eval"))


def _clip(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + "…"


def _summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep the trace small; the model still sees the full observation."""
    if not observation.get("ok", True):
        return {"ok": False, "error": observation.get("error", "未知错误")}
    summary: dict[str, Any] = {"ok": True}
    if "results" in observation:
        results = observation["results"]
        summary["count"] = len(results)
        summary["hits"] = [
            {"paper_id": row.get("paper_id"), "page": row.get("page_number"),
             "score": round(float(row.get("score") or 0), 4),
             "preview": _clip(row.get("text"), 120)}
            for row in results[:3]
        ]
    elif "text" in observation:
        summary["paper_id"] = observation.get("paper_id")
        summary["page"] = observation.get("page_number")
        summary["chars"] = len(observation.get("text") or "")
    elif "comparison" in observation:
        summary["papers"] = len(observation.get("comparison") or [])
    elif "fields" in observation:
        summary["fields"] = list((observation.get("fields") or {}).keys())
        summary["source"] = observation.get("source")
    elif "result" in observation:
        summary["result"] = observation.get("result")
    elif "papers" in observation:
        summary["papers"] = len(observation.get("papers") or [])
    return summary


class AgentRunner:
    def __init__(self, config: AgentConfig, db_path: Path | str, vectors_dir: Path | str,
                 chat_fn: Optional[Callable[[list[dict[str, str]]], str]] = None,
                 enable_vectors: bool = True) -> None:
        self.config = config
        self.db_path = Path(db_path)
        self.vectors_dir = Path(vectors_dir)
        self.enable_vectors = enable_vectors
        self.chat_fn = chat_fn or self._default_chat
        self.evidence: list[dict[str, Any]] = []
        self._evidence_keys: set[tuple[Any, ...]] = set()
        self.trace: list[dict[str, Any]] = []
        self.papers: list[dict[str, Any]] = []

    def _default_chat(self, messages: list[dict[str, str]]) -> str:
        return providers.chat(self.config.provider, self.config.key, self.config.model,
                              messages, temperature=self.config.temperature, timeout=120)

    @property
    def _embedding_key(self) -> Optional[str]:
        # Embeddings always use DashScope/Qwen, never the DeepSeek chat key.
        return self.config.key if self.config.provider == "qwen" else None

    # --- tool plumbing -------------------------------------------------------------

    def _add_evidence(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            page = row.get("page_number")
            if page is None:
                continue
            text = row.get("text") or ""
            if not text.strip():
                continue
            key = (row.get("paper_id"), int(page), row.get("chunk_id"), hash(text[:300]))
            if key in self._evidence_keys:
                continue
            self._evidence_keys.add(key)
            self.evidence.append({
                "paper_id": row.get("paper_id"),
                "page_number": int(page),
                "chunk_id": row.get("chunk_id", row.get("id")),
                "text": text,
                "section": row.get("section", ""),
                "heading": row.get("heading", ""),
                "paper_title": row.get("paper_title", ""),
            })
            if len(self.evidence) > 240:
                self.evidence = self.evidence[-200:]
                self._evidence_keys = {
                    (row.get("paper_id"), int(row["page_number"]), row.get("chunk_id"), hash((row.get("text") or "")[:300]))
                    for row in self.evidence if row.get("page_number") is not None
                }

    def _papers(self) -> list[dict[str, Any]]:
        if self.papers:
            return self.papers
        connection = retrieval.connect(self.db_path)
        try:
            rows = connection.execute(
                "SELECT id, title, page_count, status FROM papers ORDER BY imported_at DESC"
            ).fetchall()
        finally:
            connection.close()
        self.papers = [dict(row) for row in rows]
        return self.papers

    def _paper_row(self, paper_id: int) -> Optional[dict[str, Any]]:
        return next((paper for paper in self._papers() if int(paper["id"]) == int(paper_id)), None)

    def _validate_paper_id(self, args: dict[str, Any]) -> Optional[int]:
        raw = args.get("paper_id")
        if raw is None:
            return self.config.paper_id
        try:
            paper_id = int(raw)
        except (TypeError, ValueError):
            raise ValueError("paper_id 必须是整数")
        if self._paper_row(paper_id) is None:
            raise ValueError(f"论文 {paper_id} 不存在")
        return paper_id

    def _execute(self, tool: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(args, dict):
            raise ValueError("args 必须是对象")
        if tool == "list_papers":
            papers = [{"id": paper["id"], "title": paper["title"],
                       "page_count": paper.get("page_count"), "status": paper.get("status")}
                      for paper in self._papers()]
            return {"ok": True, "papers": papers}, []
        if tool == "search_library":
            query = str(args.get("query") or "").strip()
            if not query:
                raise ValueError("query 不能为空")
            paper_id = self._validate_paper_id(args)
            top_k = max(1, min(int(args.get("top_k") or 8), 15))
            rows = retrieval.hybrid_search(
                query, self.db_path, self.vectors_dir, paper_id=paper_id,
                top_k=top_k, enable_vectors=self.enable_vectors,
                api_key=self._embedding_key,
            )
            self._add_evidence(rows)
            return {"ok": True, "results": [
                {"chunk_id": row["id"], "paper_id": row["paper_id"],
                 "paper_title": row.get("paper_title", ""),
                 "page_number": row["page_number"], "section": row.get("section", ""),
                 "heading": row.get("heading", ""),
                 "text": _clip(row.get("text"), 1600), "score": row.get("score")}
                for row in rows
            ]}, rows
        if tool == "read_page":
            paper_id = self._validate_paper_id(args)
            if paper_id is None:
                raise ValueError("read_page 需要 paper_id")
            try:
                page = int(args.get("page"))
            except (TypeError, ValueError):
                raise ValueError("page 必须是整数")
            try:
                row = retrieval.read_page(self.db_path, paper_id, page)
            except LookupError as error:
                raise ValueError(str(error)) from error
            evidence = [{"paper_id": paper_id, "page_number": page, "chunk_id": f"page-{paper_id}-{page}",
                         "text": row["text"], "paper_title": row.get("paper_title", "")}]
            self._add_evidence(evidence)
            return {"ok": True, "paper_id": paper_id, "page_number": page,
                    "paper_title": row.get("paper_title", ""), "text": _clip(row["text"], 8000)}, evidence
        if tool == "read_section":
            paper_id = self._validate_paper_id(args)
            if paper_id is None:
                raise ValueError("read_section 需要 paper_id")
            heading = str(args.get("heading") or "").strip()
            if not heading:
                raise ValueError("heading 不能为空")
            rows = retrieval.read_section(self.db_path, paper_id, heading)
            self._add_evidence(rows)
            return {"ok": True, "results": [
                {"chunk_id": row["id"], "paper_id": row["paper_id"], "page_number": row["page_number"],
                 "section": row.get("section", ""), "heading": row.get("heading", ""),
                 "text": _clip(row.get("text"), 1600)}
                for row in rows
            ]}, rows
        if tool == "compare_papers":
            raw_ids = args.get("paper_ids") or []
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError("paper_ids 必须是非空列表")
            try:
                paper_ids = [int(value) for value in raw_ids][:5]
            except (TypeError, ValueError):
                raise ValueError("paper_ids 必须是整数列表")
            for paper_id in paper_ids:
                if self._paper_row(paper_id) is None:
                    raise ValueError(f"论文 {paper_id} 不存在")
            aspects = args.get("aspects") or []
            if isinstance(aspects, str):
                aspects = [aspects]
            if not isinstance(aspects, list) or not aspects:
                raise ValueError("aspects 必须是非空列表")
            comparison = retrieval.compare_papers(
                self.db_path, self.vectors_dir, paper_ids, [str(item) for item in aspects],
                enable_vectors=self.enable_vectors,
                api_key=self._embedding_key,
            )
            rows: list[dict[str, Any]] = []
            for group in comparison:
                rows.extend(group.get("chunks") or [])
            self._add_evidence(rows)
            clipped = [
                {**group, "chunks": [
                    {**chunk, "text": _clip(chunk.get("text"), 800)} for chunk in group.get("chunks") or []
                ]}
                for group in comparison
            ]
            return {"ok": True, "comparison": clipped}, rows
        if tool == "extract_paper":
            paper_id = self._validate_paper_id(args)
            if paper_id is None:
                raise ValueError("extract_paper 需要 paper_id")
            from .structured import extract_fields
            analysis = extract_fields(paper_id, self.db_path, self.config.provider,
                                      self.config.key, self.config.model)
            evidence: list[dict[str, Any]] = []
            for field_value in (analysis.get("fields") or {}).values():
                for item in field_value.get("evidence") or []:
                    page = item.get("page_number")
                    if page is None:
                        continue
                    try:
                        page_row = retrieval.read_page(self.db_path, paper_id, int(page))
                    except LookupError:
                        continue
                    evidence.append({"paper_id": paper_id, "page_number": int(page),
                                     "chunk_id": f"page-{paper_id}-{page}", "text": page_row["text"],
                                     "paper_title": page_row.get("paper_title", "")})
            self._add_evidence(evidence)
            fields = {
                name: {"value": _clip(value.get("value"), 800),
                       "evidence": [
                           {"page_number": item.get("page_number"), "quote": _clip(item.get("quote"), 240)}
                           for item in (value.get("evidence") or [])[:3]
                       ]}
                for name, value in (analysis.get("fields") or {}).items()
            }
            return {"ok": True, "fields": fields, "source": analysis.get("source"),
                    "cached": analysis.get("cached", False)}, evidence
        if tool == "calculate":
            expression = str(args.get("expression") or "").strip()
            result = _safe_calculate(expression)
            return {"ok": True, "expression": expression, "result": round(result, 6)}, []
        raise ValueError(f"未知工具：{tool}")

    # --- finalisation --------------------------------------------------------------

    def _repair_citations(self, messages: list[dict[str, str]], finish_call: dict[str, Any],
                          validation: Any) -> Optional[dict[str, Any]]:
        invalid = validation.invalid[:5]
        if not invalid:
            return None
        repair = (
            "你上一条 finish 的以下引用无法在工具证据原文中核对：\n"
            + json.dumps(invalid, ensure_ascii=False)
            + "\n请只输出一个 JSON 对象，tool 为 finish。citations 中的 quote 必须逐字复制自"
            "工具 Observation 中出现过的原文，page 必须与证据页码一致。不要改写 quote，不要新增没有证据的引用。"
        )
        try:
            raw = self.chat_fn([*messages,
                                {"role": "assistant", "content": json.dumps(finish_call, ensure_ascii=False)[:5000]},
                                {"role": "user", "content": repair}])
        except providers.ProviderError:
            return None
        call = providers.extract_json_object(raw)
        if call and call.get("tool") == "finish" and isinstance(call.get("args"), dict):
            return call
        return None

    def _fallback(self, reason: str) -> tuple[str, list[dict[str, Any]]]:
        rows = [row for row in self.evidence if row.get("text", "").strip()][:6]
        if not rows:
            return "", []
        lines = [f"未能生成可核验的模型回答（{reason}）。以下是从本地论文中检索到的相关原文，供你自行核对："]
        citations: list[dict[str, Any]] = []
        for row in rows:
            quote = row["text"].strip()[:240]
            lines.append(f"- [Paper {row.get('paper_id')} · Page {row['page_number']}] {quote}")
            citations.append({"paper_id": row.get("paper_id"), "page": int(row["page_number"]),
                              "quote": quote, "claim": "检索到的相关原文", "verified": True, "level": "quote"})
        return "\n".join(lines), citations

    def _finish(self, args: dict[str, Any], messages: list[dict[str, str]],
                question: str) -> tuple[str, list[dict[str, Any]], bool, list[str], str]:
        answer = str(args.get("answer") or "").strip()
        raw_citations = args.get("citations")
        if not isinstance(raw_citations, list):
            raw_citations = []
        require_citation = self.config.mode == "paper"
        validation = validate_citations(
            answer, raw_citations, self.evidence,
            default_paper_id=self.config.paper_id, require_citation=require_citation,
        )
        repaired = False
        if not validation.ok and validation.invalid and not repaired:
            finish_call = {"thought": "修正引用", "tool": "finish", "args": args}
            repaired_call = self._repair_citations(messages, finish_call, validation)
            if repaired_call:
                repaired = True
                answer = str(repaired_call["args"].get("answer") or answer).strip()
                repaired_citations = repaired_call["args"].get("citations")
                validation = validate_citations(
                    answer, repaired_citations if isinstance(repaired_citations, list) else raw_citations,
                    self.evidence, default_paper_id=self.config.paper_id,
                    require_citation=require_citation,
                )
        warnings = list(validation.warnings)
        verified = validation.ok and (validation.level != "none" or not require_citation)
        if require_citation and not any(item.verified for item in validation.citations):
            fallback_answer, fallback_citations = self._fallback("引用无法核对")
            if fallback_answer:
                answer = fallback_answer
                validation = validate_citations(
                    answer, fallback_citations, self.evidence,
                    default_paper_id=self.config.paper_id, require_citation=True,
                )
                warnings = ["模型回答缺少可核验引用，已改为展示检索原文。"]
                verified = any(item.verified for item in validation.citations)
            else:
                answer = "当前论文中没有检索到足以回答该问题的证据。请换一种问法，或先确认 PDF 已成功解析。"
                warnings.append("没有可用证据。")
                verified = False
        if not answer:
            fallback_answer, fallback_citations = self._fallback("模型没有给出回答")
            if fallback_answer:
                answer = fallback_answer
                validation = validate_citations(
                    answer, fallback_citations, self.evidence,
                    default_paper_id=self.config.paper_id, require_citation=require_citation,
                )
                verified = any(item.verified for item in validation.citations) or not require_citation
            else:
                raise AgentError("模型没有返回可用回答")
        if verified and validation.citations and not PAGE_RE.search(answer):
            answer = answer + "\n\n" + citation_footer(validation.citations)
        answer, _ = apply_safety(answer, question)
        return answer, validation.as_dicts(), verified, warnings, validation.level

    # --- persistence ---------------------------------------------------------------

    def _persist(self, question: str, answer: str, citations: list[dict[str, Any]],
                 verified: bool, level: str, warnings: list[str]) -> Optional[int]:
        try:
            connection = retrieval.connect(self.db_path)
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS agent_runs ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id INTEGER, mode TEXT NOT NULL,"
                    "question TEXT NOT NULL, answer TEXT NOT NULL, provider TEXT, model TEXT,"
                    "verified INTEGER NOT NULL DEFAULT 0, citation_level TEXT,"
                    "trace_json TEXT NOT NULL DEFAULT '[]', citations_json TEXT NOT NULL DEFAULT '[]',"
                    "warnings_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS agent_steps ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,"
                    "step_index INTEGER NOT NULL, thought TEXT, tool TEXT, args_json TEXT,"
                    "observation_json TEXT, duration_ms INTEGER, created_at TEXT NOT NULL)"
                )
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                cursor = connection.execute(
                    "INSERT INTO agent_runs(paper_id,mode,question,answer,provider,model,verified,"
                    "citation_level,trace_json,citations_json,warnings_json,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.config.paper_id, self.config.mode, question, answer,
                     self.config.provider, self.config.model, 1 if verified else 0, level,
                     json.dumps(self.trace, ensure_ascii=False), json.dumps(citations, ensure_ascii=False),
                     json.dumps(warnings, ensure_ascii=False), now),
                )
                run_id = int(cursor.lastrowid)
                for step in self.trace:
                    connection.execute(
                        "INSERT INTO agent_steps(run_id,step_index,thought,tool,args_json,"
                        "observation_json,duration_ms,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (run_id, step.get("index"), step.get("thought"), step.get("tool"),
                         json.dumps(step.get("args"), ensure_ascii=False),
                         json.dumps(step.get("observation"), ensure_ascii=False),
                         step.get("duration_ms"), now),
                    )
                connection.commit()
                return run_id
            finally:
                connection.close()
        except sqlite3.Error:
            return None

    # --- public entry points -------------------------------------------------------

    def _initial_rows(self, question: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            if self.config.mode == "paper" and self.config.paper_id is not None:
                rows = retrieval.hybrid_search(
                    question, self.db_path, self.vectors_dir,
                    paper_id=self.config.paper_id, top_k=6,
                    enable_vectors=self.enable_vectors,
                    api_key=self._embedding_key,
                )
            elif self.config.mode == "library":
                rows = retrieval.library_search(
                    question, self.db_path, self.vectors_dir, top_k=6,
                    enable_vectors=self.enable_vectors,
                    api_key=self._embedding_key,
                )
        except Exception:
            rows = []
        self._add_evidence(rows)
        return rows

    def run_iter(self, question: str) -> Iterator[dict[str, Any]]:
        question = (question or "").strip()
        if not question:
            yield {"type": "error", "error": "问题不能为空"}
            return
        assessment = assess_question(question)
        yield {"type": "start", "mode": self.config.mode, "paper_id": self.config.paper_id,
               "safety": {"emergency": assessment.emergency, "matched": assessment.matched}}
        papers = self._papers()
        if self.config.mode == "paper" and self.config.paper_id is not None:
            if self._paper_row(self.config.paper_id) is None:
                yield {"type": "error", "error": "论文不存在"}
                return
        initial_rows = self._initial_rows(question)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _system_prompt(self.config)},
            {"role": "user", "content": _initial_user_prompt(self.config, question, papers, initial_rows)},
        ]

        for step_index in range(1, self.config.max_steps + 1):
            started = time.time()
            try:
                raw = self.chat_fn(messages)
            except providers.ProviderError as error:
                fallback_answer, fallback_citations = self._fallback(str(error))
                if fallback_answer and self.config.mode in ("paper", "library"):
                    validation = validate_citations(
                        fallback_answer, fallback_citations, self.evidence,
                        default_paper_id=self.config.paper_id, require_citation=False,
                    )
                    answer, _ = apply_safety(fallback_answer, question)
                    run_id = self._persist(question, answer, validation.as_dicts(), True,
                                           validation.level, [str(error)])
                    yield {"type": "done", "id": run_id, "content": answer,
                           "citations": validation.as_dicts(),
                           "sources": retrieval.to_sources(self.evidence[:12]),
                           "trace": self.trace, "verified": True,
                           "citation_level": validation.level,
                           "warnings": [f"模型服务异常，已回退到检索原文：{error}"],
                           "safety": {"emergency": assessment.emergency, "matched": assessment.matched}}
                else:
                    yield {"type": "error", "error": str(error)}
                return

            call = providers.extract_json_object(raw)
            if call is None:
                observation = {"ok": False, "error": "输出不是合法 JSON，请只输出一个 JSON 对象"}
                step = {"index": step_index, "thought": "", "tool": "invalid_json",
                        "args": {}, "observation": observation,
                        "duration_ms": int((time.time() - started) * 1000)}
                self.trace.append(step)
                yield {"type": "step", **step}
                messages.append({"role": "assistant", "content": _clip(raw, 2000)})
                messages.append({"role": "user", "content":
                                 "上一条输出不是合法 JSON。请只输出一个 JSON 对象，"
                                 "格式为 {\"thought\":...,\"tool\":...,\"args\":{...}}。"})
                continue

            thought = str(call.get("thought") or "")
            tool = str(call.get("tool") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if tool not in TOOL_SPECS:
                observation = {"ok": False, "error": f"未知工具：{tool}"}
                step = {"index": step_index, "thought": thought, "tool": tool or "unknown",
                        "args": args, "observation": observation,
                        "duration_ms": int((time.time() - started) * 1000)}
                self.trace.append(step)
                yield {"type": "step", **step}
                messages.append({"role": "assistant", "content": _clip(json.dumps(call, ensure_ascii=False), 3000)})
                messages.append({"role": "user", "content": "Observation:\n" + json.dumps(observation, ensure_ascii=False)})
                continue

            if tool == "finish":
                try:
                    answer, citations, verified, warnings, level = self._finish(args, messages, question)
                except AgentError as error:
                    yield {"type": "error", "error": str(error)}
                    return
                step = {"index": step_index, "thought": thought, "tool": "finish", "args": {
                    "answer": _clip(args.get("answer"), 500),
                    "citation_count": len(args.get("citations") or []) if isinstance(args.get("citations"), list) else 0,
                }, "observation": {"ok": True, "verified": verified, "citation_level": level},
                    "duration_ms": int((time.time() - started) * 1000)}
                self.trace.append(step)
                yield {"type": "step", **step}
                run_id = self._persist(question, answer, citations, verified, level, warnings)
                yield {"type": "done", "id": run_id, "content": answer, "citations": citations,
                       "sources": retrieval.to_sources(self.evidence[:12]),
                       "trace": self.trace, "verified": verified, "citation_level": level,
                       "warnings": warnings,
                       "safety": {"emergency": assessment.emergency, "matched": assessment.matched}}
                return

            try:
                observation, evidence = self._execute(tool, args)
                self._add_evidence(evidence)
            except (ValueError, LookupError, TypeError, ZeroDivisionError) as error:
                observation = {"ok": False, "error": str(error)}
            step = {"index": step_index, "thought": thought, "tool": tool, "args": args,
                    "observation": _summarize_observation(observation),
                    "duration_ms": int((time.time() - started) * 1000)}
            self.trace.append(step)
            yield {"type": "step", **step}
            messages.append({"role": "assistant", "content": _clip(json.dumps(call, ensure_ascii=False), 4000)})
            messages.append({"role": "user", "content":
                             "Observation:\n" + _clip(json.dumps(observation, ensure_ascii=False), MAX_OBSERVATION_CHARS)})

        # Max steps reached: force a final answer with the evidence already gathered.
        messages.append({"role": "user", "content":
                         "已达到最大工具步数。请立刻调用 finish，基于已有证据回答；"
                         "证据不足的部分明确说明，不要编造引用。"})
        started = time.time()
        try:
            raw = self.chat_fn(messages)
            call = providers.extract_json_object(raw)
        except providers.ProviderError:
            call = None
        if call and call.get("tool") == "finish" and isinstance(call.get("args"), dict):
            try:
                answer, citations, verified, warnings, level = self._finish(call["args"], messages, question)
            except AgentError as error:
                yield {"type": "error", "error": str(error)}
                return
            step = {"index": self.config.max_steps + 1, "thought": str(call.get("thought") or ""),
                    "tool": "finish", "args": {"answer": _clip(call["args"].get("answer"), 500)},
                    "observation": {"ok": True, "verified": verified, "citation_level": level},
                    "duration_ms": int((time.time() - started) * 1000)}
            self.trace.append(step)
            yield {"type": "step", **step}
            run_id = self._persist(question, answer, citations, verified, level, warnings)
            yield {"type": "done", "id": run_id, "content": answer, "citations": citations,
                   "sources": retrieval.to_sources(self.evidence[:12]),
                   "trace": self.trace, "verified": verified, "citation_level": level,
                   "warnings": warnings,
                   "safety": {"emergency": assessment.emergency, "matched": assessment.matched}}
            return
        fallback_answer, fallback_citations = self._fallback("达到最大步数")
        if fallback_answer:
            validation = validate_citations(fallback_answer, fallback_citations, self.evidence,
                                            default_paper_id=self.config.paper_id, require_citation=False)
            answer, _ = apply_safety(fallback_answer, question)
            run_id = self._persist(question, answer, validation.as_dicts(), True, validation.level,
                                   ["达到最大工具步数，已回退到检索原文。"])
            yield {"type": "done", "id": run_id, "content": answer,
                   "citations": validation.as_dicts(),
                   "sources": retrieval.to_sources(self.evidence[:12]),
                   "trace": self.trace, "verified": True, "citation_level": validation.level,
                   "warnings": ["达到最大工具步数，已回退到检索原文。"],
                   "safety": {"emergency": assessment.emergency, "matched": assessment.matched}}
        else:
            yield {"type": "error", "error": "达到最大步数且没有可用证据"}

    def run(self, question: str) -> dict[str, Any]:
        done: Optional[dict[str, Any]] = None
        for event in self.run_iter(question):
            if event.get("type") == "error":
                raise AgentError(event.get("error") or "Agent 运行失败")
            if event.get("type") == "done":
                done = event
        if done is None:
            raise AgentError("Agent 没有返回结果")
        return done
