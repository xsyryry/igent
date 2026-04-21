# IELTS Study Assistant Agent

一个面向本地演示与后续迭代的雅思学习助手 Agent Demo。

当前项目采用单 Agent、CLI、本地 SQLite、可选 LLM、内置 simple RAG 的 MVP 架构，重点是：
- 主流程清晰可跑
- 检索、批改、记忆三条链路闭环
- 不依赖外部 RAG 服务也能本地运行

## 项目简介

当前项目已经具备以下能力：
- CLI 对话入口
- LangGraph 风格主流程
- Router / Planner / Tool Executor / Context Builder / Generator / Memory Writer 节点
- SQLite 用户画像、学习计划、错题记录
- 本地 simple RAG 检索
- Gap-driven retrieval 多轮检索评测
- LLM 驱动的智能路由、规划、回答生成、自动分块（可选）
- Calendar / Web Search 工具的真实接口预留与 mock fallback

## 当前架构

```text
project/
├─ app/
│  ├─ main.py
│  ├─ chunk_preview.py
│  ├─ chunk_eval.py
│  ├─ rag_uploader.py
│  ├─ retrieval_eval.py
│  └─ writing_data_manager.py
├─ agent/
├─ db/
├─ llm/
├─ memory/
├─ rag/
│  ├─ chunking.py
│  ├─ chunking_agent.py
│  ├─ ingestion_plan.py
│  ├─ local_index.py
│  └─ simple_rag.py
├─ retrieval/
│  ├─ gap_retrieval.py
│  ├─ novelty_ranker.py
│  └─ eval.py
├─ tools/
├─ writing/
├─ data/
├─ config.py
├─ requirements.txt
└─ .env.example
```

## Agent Skills

- `project/agent/skills/chunking_skill.md`：指导 LLM 选择资料分块策略。
- `project/agent/skills/get_data_skill.md`：指导 Agent 收集、下载、归档公开雅思/RAG 资料。

## 环境变量

### 基础配置

```bash
APP_NAME=IELTS Study Assistant
LOG_LEVEL=INFO
DEFAULT_TOP_K=5
DEFAULT_DATASET_SCOPE=ielts_core
RAG_BACKEND=simple_local
```

### LLM 配置

```bash
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
CHUNK_LLM_MODEL=
LLM_TIMEOUT=30
```

### Retrieval 配置

```bash
RETRIEVAL_MAX_ROUNDS=3
RETRIEVAL_MAX_NO_PROGRESS_ROUNDS=2
RETRIEVAL_TOP_K_PER_ROUND=5
RETRIEVAL_SELECTED_K=3
RETRIEVAL_DUPLICATE_RATE_THRESHOLD=0.75
RETRIEVAL_GAP_FILL_TARGET=0.75
RETRIEVAL_NOVELTY_WEIGHT=0.35
RETRIEVAL_HISTORY_PENALTY=0.4
RETRIEVAL_MIN_NEW_FACTS=1
```

### 数据层 / 外部工具

```bash
DB_BACKEND=mock_sqlite
CALENDAR_BACKEND=mock_calendar
GOOGLE_CALENDAR_CREDENTIALS=
GOOGLE_CALENDAR_ID=
SEARCH_BACKEND=mock_search
TAVILY_API_KEY=
```

## 安装与运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r project/requirements.txt
python -m project.app.main
```

## 本地 simple RAG

当前项目不再依赖外部 RAG 服务。

- `project/rag/simple_rag.py`
  - 优先读取 `data/rag_index/` 的持久化索引
  - 未建立索引时才临时扫描 `data/` 和 `project/data/`
- `project/rag/local_index.py`
  - 保存 document / paragraph chunk / sentence chunk / metadata / lexical vector
- `project/tools/rag_tool.py`
  - 对外统一暴露 `retrieve_knowledge(...)`
- `project/rag/orchestration/gap_retrieval.py`
  - 多轮检索、缺口驱动检索、新颖性排序

### 持久化索引

```bash
python -m project.app.rag_indexer build
python -m project.app.rag_indexer build --publication economist --date-from 2026.01.01 --date-to 2026.01.31
python -m project.app.rag_indexer build --magazine wired --date-from 2026-01-01 --date-to 2026-03-31
python -m project.app.rag_indexer status
python -m project.app.rag_indexer clear
```

默认数据根为 `data/awesome-english-ebooks`，只索引每期期刊目录下的 PDF 文件，忽略 `.epub` 和 `.mobi`。

索引方法：
- 文档级：记录 source、标题、hash、mtime、分块策略
- 段落级：按 headings / qa_pairs / rubric_items / mistake_rules / sliding 生成主 chunk
- 句子级：从主 chunk 中抽取高价值 cause-effect / concession / example 等句子
- 元数据：保存 publication、issue_date、topic、paragraph_role、stance、register、sentence_pattern、keywords、entities、visibility、owner_id
- 检索：读取持久化 chunks，支持 scope/filter/用户私有资料过滤和跨轮去重

## 资料准备与分块

### 分块预览

```bash
python -m project.app.chunk_preview --file academic-test-sample-questions.html --strategy headings
python -m project.app.chunk_preview --file ielts-writing-band-descriptors.pdf --strategy rubric_items
python -m project.app.chunk_preview --file academic-test-sample-questions.html --strategy llm_auto
```

### 本地 RAG 资料准备

```bash
python -m project.app.rag_uploader --tier tier1 --dry-run
python -m project.app.rag_uploader --tier tier1
python -m project.app.rag_uploader --file academic-test-sample-questions.html --use-chunks --chunk-strategy headings
python -m project.app.rag_uploader --file academic-test-sample-questions.html --use-chunks --chunk-strategy llm_auto
```

说明：
- 该命令现在只做本地资料准备，不再上传外部 RAG 服务
- `--use-chunks` 会导出 chunk preview JSONL，方便人工检查
- 建议准备资料后运行 `python -m project.app.rag_indexer build --publication economist --date-from 2026.01.01 --date-to 2026.01.31` 固化索引

## 分块评测

```bash
python -m project.app.chunk_eval
python -m project.app.chunk_eval --strategies sliding,headings,llm_auto
python -m project.app.chunk_eval --strategies llm_auto --show-cases
```

## 多轮检索评测

```bash
python -m project.app.retrieval_eval
python -m project.app.retrieval_eval --limit 1
```

## 写作批改与错题本

当前写作批改链路：

```text
用户提交作文
-> Router 识别为写作批改
-> Planner 进入 writing review
-> gap retrieval 多轮检索
-> finalize_task2_review
-> SQLite 写入 submission + mistake record
-> 刷新用户画像中的 mistake_patterns / focus_recommendations
```

## Memory

当前 memory 分成两层：
- Session context
  - `messages`
  - `study_context`
- Persistent memory（SQLite）
  - `users`
  - `study_plans`
  - `mistake_records`

## 当前真实实现 / fallback

### 当前真实实现
- SQLite 数据层
- CLI 与主流程编排
- 本地 simple RAG
- Gap-driven retrieval
- 写作批改 + 错题本

### 当前带 fallback 的能力
- LLM 路由 / 规划 / 生成 / 自动分块
- Calendar tool
- Web search tool

## 后续方向

- 扩充本地资料库与评测集
- 继续优化 auto chunking 稳定性
- 扩展写作/口语批改
- 增加文件上传/OCR 版错题导入
