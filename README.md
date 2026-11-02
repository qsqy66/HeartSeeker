# 觅心（HeartSeeker_Med）— 本地优先的医学论文 Agentic RAG

面向医学生和临床研究者的论文精读助手。导入 PDF 后，在本机完成解析、混合检索与可核验问答；数据默认只存本机，只有主动使用模型功能时才向远端供应商发送必要的文本片段。

问答与嵌入仍使用远端模型；检索索引、原文、向量和问答历史全部落在本机。OCR、重排和论断蕴含校验预留了可选本地模型接口，默认使用无需模型的本地词法/规则实现。问答模型支持 OpenAI、DeepSeek、Qwen、Anthropic Claude、Google Gemini、Moonshot、智谱、SiliconFlow、OpenRouter、Groq、Mistral、Ollama 以及自定义网关；供应商按 **OpenAI 兼容 / Anthropic Messages / Gemini** 三种协议适配，新增同协议厂商只需增加配置。

语义检索嵌入仍固定使用千问 DashScope 兼容接口（默认 `text-embedding-v3`），向量存本机。问答供应商与嵌入供应商的 Key 分开保存，因此可以把 OpenAI 作为问答模型、同时用千问计算嵌入。

## 能力概览

### Agentic RAG

- 论文内问答、全库问答、通用医学对话统一走 `POST /api/agent/chat`。
- 模型按 JSON ReAct 协议多步调用工具：`list_papers`、`search_library`、`read_page`、`read_section`、`compare_papers`、`extract_paper`、`calculate`、`clinical_calculator`、`lookup_ontology`、`appraise_evidence`、`evidence_table`、`remember_note`、`recall_notes`、`forget_note`、`finish`。
- 每次运行记录 `thought / tool / args / observation / duration`，前端展示可展开的执行轨迹。
- 自动模式：带 `paper_id` 时限定当前论文；不带时在已有论文库中检索；库为空时走通用医学知识。
- 最大步数、非法 JSON 修正、工具异常降级、无证据兜底均有明确处理。
- 支持 SSE/NDJSON 流式回答：前端在 `finish.answer` 生成过程中增量渲染，并可随时点击「停止」中止请求。
- 长对话按 `context_token_budget` 估算并裁剪旧步骤，始终保留系统提示和原始问题。
- 本地长期记忆存放在 SQLite（`agent_memories`），只保存用户明确要求记住的内容，可随时删除；问答前按词项重叠召回相关笔记。
- 运行结果可导出为 Markdown（回答 + 引用 + 轨迹）、JSON（完整记录）或 CSV（仅引用表）。

### 多供应商与协议适配

- 供应商只是配置，协议决定请求/响应格式；同一个 Agent、结构化精读和追问链路可切换不同厂商。
- `openai`：OpenAI 兼容 Chat Completions，支持 Bearer Key、`/models` 列表和 SSE 流式增量。
- `anthropic`：Messages API，使用 `x-api-key` + `anthropic-version`，流式解析 `content_block_delta`。
- `gemini`：`generateContent` / `streamGenerateContent`，使用 `x-goog-api-key`，支持 system instruction 和模型列表过滤。
- 「自定义」供应商可选择协议并填写 Base URL；内置供应商也允许覆盖 Base URL，方便代理或自建网关。
- Ollama / 自定义本地网关可以留空 API Key；需要 Key 的供应商会给出明确提示。
- 问答 Key 与 DashScope 嵌入 Key 分开存储；切换问答供应商不会把其 Key 发送到嵌入接口。

### 可核验引用

- 最终回答必须给出 `{paper_id, page, quote, claim}`；`quote` 必须逐字出现在该页原文中。
- 校验前做 NFKC、断行连字符、空白与标点归一化；页码不存在或引文不匹配即判定失败。
- 首次引用失败会请求模型修正一次；论文模式仍无可核验引用时，回退为直接展示检索原文，不保留模型结论。
- 在逐字引文之外做论断级核验：数字/单位一致性、否定词方向、中英文混合匹配；可选本地 NLI cross-encoder 判定 `quote` 是否支持 `claim`，不可用时自动回退到规则判定。
- 前端把 `[Page N]` / `[Paper N · Page M]` 渲染为可点击链接，点击回到 PDF 对应页；有 bbox 的证据会高亮原文区块。

### 混合检索

- SQLite FTS5（BM25 排序）+ LanceDB 向量召回，用 Reciprocal Rank Fusion 融合两路排名。
- 中文分词：优先使用 jieba，未安装时退化为 bigram；FTS5 索引使用分词后的文本，兼顾中英文查询。
- 英文医学缩写与中文词表双向扩展（MI / myocardial infarction / 心梗，HF / heart failure / 心衰……）。
- 中文问题额外做疑问词切分与短 n-gram 回退，解决 FTS5 对中文不分词、整句 LIKE 匹配不到的问题。
- 可插拔重排：默认本地词法重排（无需模型），可切换本地 cross-encoder 或 Qwen 远程重排（默认 `qwen3.7-text-rerank`，Base URL、模型和 Key 均可在设置页填写）；调用失败时自动退回词法重排，不中断检索。
- 支持单篇论文检索、全库检索、按章节阅读和跨论文对比；父子块上下文可扩展到同章节相邻片段。

### 医学安全与结构化精读

- 急症红旗识别（胸痛、呼吸困难、意识障碍、卒中、严重出血、过敏、自伤、高热等），回答前加入就医提示（中国大陆 120）。
- 所有医学回答附带学习用途声明；检索到的文档用不可信数据分隔符包裹，降低提示注入风险。
- 结构化字段：一句话总结、研究设计、人群、干预/暴露、对照、结局、样本量、统计方法、关键结果、局限性、证据等级。
- LLM 提取的每个字段都要求原文引文并逐字校验；支持 SQLite 缓存，未配置 Key 时回退到规则提取。
- 页面中无明确证据的字段统一标注「论文未明确报告」。

### 本地 OCR（扫描件）

- 页面没有文字层时，可调用本地 OCR：优先 RapidOCR（`rapidocr-onnxruntime`），其次 Tesseract（系统命令或 `TESSERACT_BIN`）。
- 也可选择阿里云 OCR 统一识别（`RecognizeAllText`，SDK：`alibabacloud_ocr_api20210707`）。设置页集中提供 AccessKey ID / Secret、Endpoint、识别类型和连接测试；Secret 只存入系统凭据库，AccessKey ID 与 Endpoint 存在本地设置数据库。
- 设置页的「测试阿里云 OCR 连接」会渲染一张测试图片并调用一次 `RecognizeAllText`，用于在导入论文前验证凭据、Endpoint 和权限，会消耗一次 OCR 调用。
- AccessKey 需为 RAM 用户，授予 `AliyunOCRFullAccess`，或按最小权限只授予 `ocr:RecognizeAllText`：

  ```json
  {
    "Version": "1",
    "Statement": [
      { "Effect": "Allow", "Action": "ocr:RecognizeAllText", "Resource": "*" }
    ]
  }
  ```

  建议只使用程序访问并定期轮换。

- 也可以改用环境变量保存 AccessKey（可选），此时设置页的 AccessKey ID / Secret 留空：

  ```bash
  # Linux / macOS（写入 shell 配置后重新打开终端）
  export ALIBABA_CLOUD_ACCESS_KEY_ID="<access_key_id>"
  export ALIBABA_CLOUD_ACCESS_KEY_SECRET="<access_key_secret>"

  # Windows CMD（永久，需新开窗口）
  setx ALIBABA_CLOUD_ACCESS_KEY_ID "<access_key_id>"
  setx ALIBABA_CLOUD_ACCESS_KEY_SECRET "<access_key_secret>"
  ```

  若设置页也填写了凭据，则优先使用设置页的值（Secret 存在系统凭据库）；两者都未配置时阿里云 OCR 不可用。
- Endpoint 填公网接入地址（如 `ocr-api.cn-hangzhou.aliyuncs.com`）或同地域 VPC 地址（`ocr-api-vpc.cn-hangzhou.aliyuncs.com`）。SDK 会自动完成 RPC HMAC-SHA1 签名，无需手工拼签名参数。
- 阿里云 OCR 会将被识别页面的 PNG 上传到云端并按调用量计费。图像需小于 10MB、宽高 15–8192 像素、长宽比小于 50；超限页面会自动降低渲染 DPI，仍超限则该页记录 OCR 失败。
- 引擎选择、语言（默认 `chi_sim+eng`）和渲染 DPI（默认 220）可在设置页调整，也可用 `HEARTSEEKER_MED_OCR_ENGINE` / `HEARTSEEKER_MED_OCR_LANG` / `HEARTSEEKER_MED_OCR_DPI` 覆盖；阿里云 Endpoint/类型可用 `HEARTSEEKER_MED_ALIYUN_OCR_ENDPOINT` / `HEARTSEEKER_MED_ALIYUN_OCR_TYPE` 覆盖。
- 未安装或未配置引擎时不会失败：页面保留 `needs_ocr` 标记，文字版 PDF、检索和问答不受影响。
- OCR 结果进入正常切片和索引流程，后续检索与引用校验与文字版 PDF 一致。

### 本地医学知识工具

- 临床评分：CHA₂DS₂-VASc、HAS-BLED、CURB-65、Wells DVT、Wells PE、CKD-EPI 2021 eGFR；纯本地确定性计算，输入校验与错误提示完整。
- 本地医学本体：约 40 个中英双语概念，含别名、ICD-10、分类与相关概念；支持按名称、缩写或编码查询，可用于查询扩展。
- GRADE 启发式证据评价：识别研究设计，检查偏倚风险、不一致性、间接性、精确性与发表偏倚，输出确定性等级与理由；支持直接评价文本或本地论文。
- 以上工具同时作为 Agent 工具和独立 API 暴露，不依赖模型即可运行。

### 工程与体验

- PDF 解析在单线程后台任务中执行，导入接口立即返回，前端轮询 `progress_stage / progress_current / progress_total`。
- 支持失败重试与「重建索引」（重新嵌入已有切片，不重新解析 PDF）。
- SQLite WAL、`busy_timeout`、外键与增量迁移；旧数据库可直接启动。
- 前端支持 Markdown/表格渲染、执行轨迹、核验状态、警告、全库检索、导入进度和引用跳转。
- API Key 存入操作系统凭据库（keyring），不落论文数据库、不写日志。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 19 + TypeScript + Vite + pdfjs-dist + react-markdown/GFM |
| 后端 | FastAPI + PyMuPDF + SQLite(FTS5) + LanceDB |
| 问答模型 | OpenAI / Anthropic / Gemini 协议及多家 OpenAI 兼容供应商（远端，可选） |
| 嵌入模型 | Qwen DashScope `text-embedding-v3`（远端计算，向量存本机） |
| 本地可选模型 | RapidOCR / Tesseract、cross-encoder 重排、cross-encoder NLI（均预留插件接口，缺省回退）；OCR 另可选阿里云 RecognizeAllText |
| 测试/评测 | Python `unittest` + 检索/引用/答案级 golden set |

> 语义检索和向量问答需要 Qwen（DashScope）Key；其他问答供应商不提供 DashScope 嵌入。没有嵌入 Key 时自动退化为关键词检索，导入、阅读和关键词问答仍可用。OpenAI、Gemini 等也可以用于问答，但不会把各自 Key 发往 DashScope。

## 环境要求

- Python 3.10+（推荐 3.10–3.12）
- Node.js 18+
- Windows 上 API Key 存储依赖系统凭据管理器

## 运行

### 自动安装（全新机器）

**Windows**：双击 `install.bat`（winget 安装 Python 3.12 + Node.js LTS）

**Linux**：`./install.sh`（自动识别 apt / dnf / yum / pacman）

> macOS 暂未提供自动安装脚本，请手动安装 Python/Node 后使用一键启动。

### 一键启动

```bash
python3 start.py        # Windows: start.bat / python start.py
```

首次运行会创建 `.venv`、安装后端依赖和前端依赖；之后复用。默认启动后端 `127.0.0.1:8765` 与前端 `127.0.0.1:5173` 并打开浏览器。`Ctrl+C` 一次停止两个服务。

可选参数：`--no-browser`；环境变量 `HEARTSEEKER_MED_PORT` / `HEARTSEEKER_MED_FRONTEND_PORT` 可覆盖端口。

### 手动启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

另开终端：

```bash
npm install
npm run dev
```

访问 <http://localhost:5173>。后端只监听 `127.0.0.1`。

## 使用流程

1. **配置模型**：设置与隐私 → 选择供应商（或「自定义」并选择协议、填写 Base URL）→ 填 API Key → 读取模型或手动填写模型 ID → 保存。语义检索另需配置 DashScope 嵌入 Key。同一页面可配置本地处理管线：OCR 引擎/语言/DPI、重排引擎与模型、NLI 模型、上下文 token 预算。
2. **导入论文**：论文库 → 导入带文本层的 PDF（≤50MB、≤300 页）。上传后立即返回，论文库卡片显示解析/向量化进度；完成后自动打开工作台。
3. **全库检索**：论文库 · 全库检索输入问题或术语，结果带论文标题、页码和混合检索来源，点击直接跳转并高亮。
4. **论文内问答**：工作台 → 论文内问答。展开「执行轨迹」可查看检索/翻页/计算等步骤和引用核验状态。
5. **通用对话**：主对话 · 通用医学；可新建多个会话，各自保留独立记忆。有论文时可切换「论文库检索」或「通用医学知识」。
6. **结构化精读**：查看带原文证据的字段；「刷新精读」会调用当前问答模型重新提取。

### Qwen 远程重排配置

设置与隐私 → 本地处理管线 → 重排引擎选择「Qwen 远程重排」，然后填写：

- **重排模型**：默认 `qwen3.7-text-rerank`，留空也会使用该默认值。
- **重排 Base URL**：默认 `https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank`，可替换为其他兼容地址。
- **重排 API Key**：单独保存在系统凭据库；留空时依次尝试千问嵌入 Key 和 `DASHSCOPE_API_KEY`。也可用环境变量 `HEARTSEEKER_MED_RERANK_URL` 覆盖默认地址。

请求按 DashScope 文本重排格式发送 `model / input.query / input.documents / parameters`。未配置 Key、网络不可用或服务报错时，检索自动退回本地词法重排，回答仍会正常生成。

## 数据与隐私

- 默认数据目录：用户目录下的 `HeartSeeker`（Windows 为 `%USERPROFILE%\HeartSeeker`），可用 `HEARTSEEKER_MED_HOME` 覆盖。
- 目录内容：`papers/`（原始 PDF）、`database/`（SQLite）、`vectors/`（LanceDB）、`cache/`、`logs/`、`exports/`。
- 远端调用边界：读取模型列表、结构化精读、Agent 问答会访问所选问答供应商；嵌入计算访问 DashScope；启用 Qwen 远程重排时会向配置的重排地址发送查询和候选片段。只发送问题、必要原文片段、嵌入文本和重排候选。问答、嵌入、重排 Key 分别保存在系统凭据库。
- Agent 运行轨迹、引用、反馈和用户笔记保存在本地 SQLite（`agent_runs`、`agent_steps`、`feedback`、`agent_memories`）。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/papers/import` | 上传 PDF，立即返回 `processing` |
| GET | `/api/papers/{id}/status` | 导入/索引进度 |
| POST | `/api/papers/{id}/retry` | 重新解析失败的论文 |
| POST | `/api/papers/{id}/reindex` | 重新嵌入已有切片 |
| GET | `/api/papers/{id}/retrieve?q=` | 单篇混合检索 |
| GET | `/api/library/search?q=&top_k=` | 全库混合检索 |
| POST | `/api/agent/chat` | Agent 问答，流式事件 `start/delta/step/done/error` |
| GET | `/api/agent/runs/{run_id}` | 运行轨迹与引用 |
| GET | `/api/agent/runs/{run_id}/export?format=md\|json\|csv` | 导出运行结果 |
| GET/POST | `/api/papers/{id}/structured` | 读取缓存/规则字段；POST 强制 LLM 提取 |
| POST | `/api/feedback` | 保存回答反馈 |
| GET/POST | `/api/memory` | 列出/新增本地记忆笔记 |
| DELETE | `/api/memory/{id}` | 删除本地记忆笔记 |
| GET | `/api/clinical/calculators` | 列出临床评分及参数 |
| POST | `/api/clinical/calculate` | 计算指定临床评分 |
| GET | `/api/ontology/lookup?q=` | 查询本地医学本体 |
| GET | `/api/ontology/stats` | 本体版本与条目统计 |
| POST | `/api/evidence/appraise` | GRADE 启发式证据评价 |
| GET/PUT | `/api/settings` | 供应商、模型、嵌入与本地管线设置 |
| GET | `/api/providers` | 可用供应商与协议列表 |
| POST | `/api/models` | 按协议读取供应商模型列表 |
| PUT/GET | `/api/settings/api-key/{provider}` | 保存/查询单个供应商 Key |

### Agent 请求示例

```bash
curl -N http://127.0.0.1:8765/api/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"这项研究的主要终点是什么？","paper_id":1,"mode":"paper","stream":true}'
```

`mode` 可为 `auto`、`paper`、`library`、`general`。事件示例：

```json
{"type":"step","index":1,"tool":"search_library","thought":"先检索主要终点","observation":{"ok":true,"count":4},"duration_ms":820}
{"type":"delta","text":"主要终点是"}
{"type":"done","content":"…… [Page 1]","citations":[{"paper_id":1,"page":1,"quote":"……","verified":true}],"verified":true,"warnings":[]}
```

`delta` 仅在模型流式生成 `finish.answer` 时出现；非 finish 步骤不会产生增量文本。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
npm run build
```

覆盖范围：中英文关键词检索、分词与词表扩展、RRF、重排、Agent 工具调用、流式 delta 解析、上下文预算、本地记忆、结果导出、引用修正与兜底、逐字引文与论断级核验、临床评分、本体查询、GRADE 评价、急症安全提示、结构化提取缓存、真实 PDF 摄取与 OCR 降级、后台状态字段、数据库迁移、离线评测，以及 OpenAI / Anthropic / Gemini 三种协议的请求头、请求体、流式解析和模型列表解析。

## 评测

默认离线评测不联网、不调用 LLM：用 `eval/corpus.json` 的三篇合成论文建立临时数据库，跑 `eval/offline_cases.json` 的 8 个中英文问题。

```bash
python scripts/evaluate.py
python scripts/evaluate.py --json
```

当前基准：

```
检索召回率: 1.0    MRR: 0.875    纯关键词召回率: 1.0
引文核验通过率: 1.0    虚构引文拒绝率: 1.0    检查数: 8
```

评测内容包括：

- Retrieval Recall@K 与 MRR（单篇 + 全库、英文 + 中文）
- 正确引文能否通过核验（正例）
- 虚构引文是否会被拒绝（负例）
- 本地 `paper_analysis` 缓存中非空字段的原文证据率

对真实论文库评测：复制 `eval/live_cases.example.json`，按格式填写 `paper_title`、`expected_pages` 和逐字引文，然后运行：

```bash
python scripts/evaluate.py \
  --db ~/HeartSeeker/database/heartseeker.sqlite3 \
  --vectors ~/HeartSeeker/vectors \
  --dataset eval/live_cases.json --no-vectors
```

`--no-vectors` 只测关键词路径；去掉并配置 Qwen Key 可测混合检索。`--min-recall` / `--min-citation` 可设置 CI 阈值，失败时退出码为 1。

### 答案级评测

`eval/answer_cases.json` 提供 8 个带预期事实与拒答场景的问题，用于评估最终回答而不只是检索命中：

```bash
# 评分已保存的回答文件（离线，不调用模型）
python scripts/evaluate_answers.py --answers eval/saved_answers.json --json

# 运行真实 Agent 并评分（未提供 --answers 时默认在线运行）
python scripts/evaluate_answers.py --provider deepseek --model deepseek-chat

# 多模型横向基准
python scripts/benchmark_models.py --provider deepseek --models deepseek-chat,qwen-plus
```

指标包括引用精确率/召回率、论断支持率、无支撑论断率、事实覆盖率、引文页召回率、拒答准确率，并支持 `--min-*` 阈值用于 CI。

## 目录结构

```
backend/
  app.py            # FastAPI 路由、流式输出、后台任务调度
  agent.py          # 多步工具调用 Agent、引用修正/兜底
  retrieval.py      # FTS5 + 向量 + RRF、中文检索、章节/对比
  rerank.py         # 可插拔本地重排（词法 / cross-encoder）
  tokenization.py   # 中英文分词（jieba / bigram）
  citations.py      # 页码/逐字引文校验
  verification.py   # 论断级核验与可选本地 NLI
  structured.py     # 结构化字段 LLM 提取、证据校验、缓存
  ingestion.py      # PDF 解析、分栏/表格、切片、向量索引
  ocr.py            # 可插拔 OCR（RapidOCR / Tesseract / 阿里云 RecognizeAllText）
  clinical.py       # 临床评分计算器
  ontology.py       # 本地医学本体查询
  evidence.py       # GRADE 启发式证据评价
  answer_eval.py    # 答案级指标与 Agent 评测
  memory.py         # 本地长期记忆
  token_budget.py   # 上下文 token 预算
  storage.py        # 路径、SQLite schema、WAL、迁移
  safety.py         # 急症提示、免责声明、不可信文档包裹
  medical_terms.py  # 中英医学词表扩展
  providers.py      # 供应商注册表 + OpenAI/Anthropic/Gemini 协议适配
  embedding.py      # DashScope 嵌入 + LanceDB
  evaluation.py     # 检索/引用离线评测
  credentials.py    # 系统凭据库
eval/
  corpus.json             # 离线合成语料
  offline_cases.json      # 离线 golden set
  answer_cases.json       # 答案级评测集
  live_cases.example.json # 真实库评测模板
scripts/
  evaluate.py             # 检索/引用评测 CLI
  evaluate_answers.py     # 答案级评测 CLI
  benchmark_models.py     # 多模型横向基准
tests/              # unittest 测试
src/
  main.tsx          # 前端应用
  styles.css
start.py            # 一键启动
```

## 已知限制

- 本地 OCR 只有在安装 RapidOCR 或 Tesseract 后才生效；阿里云 OCR 需要有效 AccessKey 和可访问的网络。两者都不可用时扫描页标记「需 OCR」。
- 表格检测依赖 PDF 边框线，无边框表格可能无法转成 Markdown。
- 问答模型与嵌入仍调用远端服务；本仓库没有内置本地 LLM 或本地嵌入模型。
- 语义检索依赖远端 DashScope 嵌入服务；服务不可用时自动退化为关键词检索。
- 记忆召回使用词项重叠打分，不是语义检索；笔记只在用户明确要求时写入。
- 本地医学本体是约 40 个概念的小型词表，临床评分和 GRADE 评价是确定性启发式，不替代正式指南或人工评价。
- 模型列表接口并非所有兼容网关都实现；读取失败时可直接手动填写模型 ID。
- 自定义供应商只有一个 Key 槽位；切换 Base URL 会复用同一把 Key。
- 医学词表是人工维护的小型映射，不替代 UMLS/SNOMED 等本体。
- 评测语料为合成数据，只验证检索与引用机制，不构成临床证据。
- 单机单用户，无账号/多租户体系。
