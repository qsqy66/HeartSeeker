# Paper Reader — 本地优先的医学论文精读智能体

面向医学生的论文阅读助手。导入论文 PDF 后，在本机完成结构化解析、语义检索与可溯源问答；数据默认只存本机，只有主动使用模型功能时才向供应商发送必要的文本片段。

## 功能特性

- **结构化 PDF 解析**：保留文本块坐标（bbox）、字体/加粗/上下标、分栏阅读顺序，章节标题自动分块，表格转 Markdown
- **可溯源检索**：语义检索（向量）+ 关键词检索（FTS5）双模式，每条结果可点击在真 PDF 上高亮对应区块
- **论文内 RAG 问答**：基于检索到的证据块回答，回答必须附带 `[Page N]` 页码引用，拒绝编造页码
- **结构化精读卡片**：PICO / 研究设计 / 样本量 / 统计方法 / 关键结果 / 局限性等字段，无证据字段标注「论文未明确报告」
- **本地优先存储**：SQLite（元数据 + 全文索引）+ LanceDB（向量），全部落在用户目录
- **隐私**：API Key 存入操作系统凭据库（Windows 凭据管理器 / keyring），不落库、不写日志

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 19 + TypeScript + Vite + pdfjs-dist |
| 后端 | FastAPI + PyMuPDF + SQLite(FTS5) + LanceDB |
| 问答模型 | DeepSeek / Qwen（可选） |
| 嵌入模型 | Qwen（DashScope `text-embedding-v3`，云端 API，向量存本机） |

## 环境要求

- Python 3.10+（推荐 3.10–3.12）
- Node.js 18+
- Windows 用户使用 API Key 存储需系统凭据管理器（桌面环境自备）

## 运行

### 1. 启动后端

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

> macOS / Linux 用 `.venv/bin/activate` 激活虚拟环境，其余命令一致。

### 2. 启动前端

```powershell
npm install
npm run dev
```

### 3. 打开

浏览器访问 <http://localhost:5173>。后端只监听 `127.0.0.1:8765`。

## 使用流程

1. **配置模型**：进入「设置与隐私」→ 选择模型供应商（DeepSeek / Qwen）→ 填入 API Key → 「读取可用模型」→ 选择模型 → 保存。
   - 仅做关键词检索和本地阅读**无需任何 Key**；
   - 启用**语义检索 / 向量问答**需要配置 **Qwen（DashScope）** 的 Key（DeepSeek 不提供嵌入接口），嵌入模型默认为 `text-embedding-v3`，可在设置中修改。
2. **导入论文**：论文库 → 导入带文本层的 PDF（≤50MB、≤300 页），自动完成解析、分块与向量化。
3. **原文阅读**：在检索框切换「语义检索 / 关键词检索」，点击右侧证据条目可在 PDF 上高亮对应位置。
4. **结构化精读**：查看自动抽取的研究字段及其原文证据。
5. **论文内问答**：提问，回答附 `[Page N]` 引用和可点击跳转的证据来源。

## 数据与隐私

- 默认数据目录：用户目录下的 `PaperReader`（Windows 为 `%USERPROFILE%\PaperReader`），可通过环境变量 `PAPER_READER_HOME` 覆盖。
- 目录内容：`papers/`（原始 PDF）、`database/`（SQLite）、`vectors/`（LanceDB 向量）、`cache/`、`logs/`、`exports/`。
- 远端调用边界：只有「读取模型列表」「结构化/问答」等主动操作才发送请求；问答只发送问题与检索到的必要原文片段。API Key 不写入论文数据库。

## 模型说明

| 用途 | 供应商 | 模型 | Key |
|---|---|---|---|
| 问答 | DeepSeek | `deepseek-chat` / 手动指定 | DeepSeek Key |
| 问答 | Qwen | `qwen-plus` / 手动指定 | Qwen Key |
| 嵌入（检索/向量问答） | Qwen（DashScope） | `text-embedding-v3`（可改） | Qwen Key |

未配置嵌入 Key 时，语义检索自动回退为 FTS5 关键词检索，应用仍可正常导入、阅读与关键词检索。

## V1 验证路径

1. 启动后端与前端，打开 <http://localhost:5173>。
2. 「设置与隐私」→ 保存 DeepSeek 和 Qwen 的 API Key。
3. 「论文库」→ 导入一篇英文医学 PDF（带文本层）。
4. 「原文阅读」→ 用「语义检索」搜 `primary outcome`，点击证据，确认 PDF 对应区块被高亮。
5. 「结构化精读」→ 查看研究设计/样本量等字段，无证据字段显示「论文未明确报告」。
6. 「论文内问答」→ 提问，确认回答包含 `[Page N]` 引用和证据来源。

## 已知限制

- 表格检测（`find_tables`）只识别带边框线的表格，无边框表格暂不检出，后续可用 `pymupdf_layout` 补强。
- 扫描版 PDF（无文字层）暂不支持 OCR，导入后标记为「需 OCR」。
- 关键词检索（FTS5）对中文分词支持有限；中文语义检索请使用「语义检索」模式（需 Qwen Key）。
- 单机部署，暂无多用户/账号体系。

## 目录结构

```
backend/
  app.py             # FastAPI 接口：导入/检索/问答/设置
  embedding.py       # DashScope 嵌入 + LanceDB 向量库封装
  requirements.txt
src/
  main.tsx           # 前端单文件应用
  styles.css
  vite-env.d.ts
```
