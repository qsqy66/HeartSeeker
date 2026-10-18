# Paper Reader V1 MVP

本仓库提供 Paper Reader V1 的本地优先实现：

- React + TypeScript 前端
- FastAPI 本地服务
- SQLite 元数据、切片与全文检索
- PyMuPDF PDF 文本解析
- 页面文本块坐标和可核验证据
- 结构化精读卡片和论文内 RAG 问答
- DeepSeek / Qwen 的显式可选调用
- Windows Credential Manager API Key 存储（通过 `keyring`）
- 不配置模型 API Key 时，仍可导入、阅读和本地检索论文

## 运行

### 本地服务

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

### 前端

```powershell
npm install
npm run dev
```

服务只监听 `127.0.0.1`。默认数据目录为用户目录下的 `PaperReader`，可通过 `PAPER_READER_HOME` 覆盖。

## V1 验证路径

1. 打开 `http://localhost:5173`。
2. 导入带文本层的英文医学 PDF。
3. 在“原文阅读”中检索关键词，点击证据跳转对应页。
4. 在“结构化精读”中查看 PICO/研究设计字段；无证据字段显示“论文未明确报告”。
5. 在“设置与隐私”中选择 DeepSeek 或 Qwen 并保存自己的 API Key。
6. 在“论文内问答”提问。远端调用前只发送问题和当前论文的检索上下文，回答必须包含 `[Page N]` 引用。

当前仓库提供浏览器版 V1 工作台。Tauri 原生安装包需要额外安装 Rust、Cargo 和 Windows WebView2 后再接入打包配置；本机未安装 Rust，因此无法在当前环境编译安装包。
