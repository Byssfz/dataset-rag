# 🏪 掌柜智库 — 基于 RAG + 知识图谱的智能产品问答系统

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-orange.svg)](https://www.langchain.com/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-3.0-blueviolet.svg)](https://milvus.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.15-brightgreen.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**掌柜智库** 是一个面向产品手册/说明书的智能检索问答（RAG）系统。支持将 PDF/Markdown 文档导入向量数据库与知识图谱，通过多路召回 + RRF 融合 + Reranker 精排 + LLM 生成，提供精准的产品知识问答。

---

## 🧭 目录

- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
  - [文档导入流程](#文档导入流程)
  - [智能查询流程](#智能查询流程)
- [技术栈](#-技术栈)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
  - [环境要求](#环境要求)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 配置环境变量](#2-配置环境变量)
  - [3. 启动依赖服务 (Docker)](#3-启动依赖服务-docker)
  - [4. 安装 Python 依赖](#4-安装-python-依赖)
  - [5. 下载模型](#5-下载模型)
  - [6. 启动服务](#6-启动服务)
- [API 接口](#-api-接口)
  - [文档导入服务 (端口 8000)](#文档导入服务-端口-8000)
  - [查询服务 (端口 8001)](#查询服务-端口-8001)
- [RAG 评估](#-rag-评估)
- [知识图谱](#-知识图谱)
- [配置说明](#-配置说明)
- [开发计划](#-开发计划)

---

## ✨ 核心特性

### 📥 智能文档导入
- **PDF 解析** — 集成 MinerU API 将 PDF 转换为 Markdown，保留表格与图片结构
- **智能分块** — 基于 Markdown 标题层级的语义分块策略
- **产品名识别** — 基于 LLM + 向量匹配自动识别文档对应的产品名称
- **BGE-M3 双向量** — 同时生成稠密向量与稀疏向量，支持混合检索
- **知识图谱抽取** — LLM 自动抽取三元组（主体-关系-客体），写入 Neo4j 图数据库

### 🔍 多路召回 + 智能排序
- **4 路并行召回**：原始向量检索 / HyDE 假设性答案检索 / 联网搜索 (MCP) / 知识图谱检索
- **RRF 融合** — Reciprocal Rank Fusion 加权融合同源/异源多路结果
- **Reranker 精排** — BGE-Reranker-Large Cross-Encoder 精确打分
- **动态 TopK 截断** — 基于断崖检测算法的自适应结果数量控制

### 💬 智能问答
- **问题重写** — 结合历史对话消除指代歧义、补全上下文
- **流式 SSE 输出** — 支持 Server-Sent Events 逐字推送答案
- **多轮对话记忆** — MongoDB 持久化历史对话，支持上下文连续问答
- **图片提取** — 自动从检索结果中提取关联图片 URL 返回前端

### 📊 RAGAS 评估
- 内置 Faithfulness / ContextRecall / FactualCorrectness / ResponseRelevancy 四项核心指标评估
- 支持自定义测试数据集，自动运行完整查询流程并生成评估报告

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端页面                              │
│            import.html  │  chat.html  │  monitor.html        │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼──────────────────────┐
    │  导入服务 (8000)     │  │   查询服务 (8001)             │
    │  FastAPI             │  │   FastAPI + SSE               │
    │  POST /upload        │  │   POST /query                 │
    │  GET  /status/{id}   │  │   GET  /stream/{session_id}   │
    └──────────┬───────────┘  │   GET  /history/{session_id}  │
               │              └───────┬───────────────────────┘
               │                      │
    ┌──────────▼──────────────────────▼───────────────────────┐
    │                   LangGraph 工作流引擎                    │
    │                                                          │
    │  导入流程:                  查询流程:                      │
    │  entry → pdf2md →           item_confirm →               │
    │  md_img → split →          ┌─ embedding_search           │
    │  item_name → bge →         ├─ hyde_search                │
    │  milvus → kg_extract       ├─ web_search (MCP)           │
    │                             └─ kg_search                  │
    │                                    ↓                      │
    │                             RRF → Rerank → Answer         │
    └──┬──────────┬──────────┬──────────┬────────────────────┘
       │          │          │          │
   ┌───▼──┐  ┌───▼───┐ ┌───▼───┐ ┌───▼────┐
   │Milvus│  │ Neo4j │ │MongoDB│ │ MinIO  │
   │向量库│  │ 图库  │ │历史记录│ │文件存储│
   └──────┘  └───────┘ └───────┘ └────────┘
```

### 文档导入流程

```
PDF文件 → [MinerU解析] → Markdown → [图片处理]
    → [语义分块] → [产品名识别] → [BGE-M3向量化]
    → [Milvus入库] → [KG三元组抽取] → [Neo4j写入]
```

### 智能查询流程

```
用户问题 → [历史对话获取] → [LLM: 问题重写 + 产品名提取]
    → [产品名向量验证]
    → ┌─ 稠密+稀疏向量检索 (Milvus)
      ├─ HyDE 假设性答案检索 (Milvus)
      ├─ 知识图谱事实召回 (Neo4j)
      └─ 联网搜索 (MCP WebSearch)       ← 四路并行
    → [RRF 加权融合排序]
    → [BGE-Reranker 精排 + 动态TopK截断]
    → [LLM 答案生成] → [SSE流式输出 / 同步返回]
    → [MongoDB 对话记录持久化]
```

---

## 🛠 技术栈

| 类别       | 技术                                                            |
| ---------- | --------------------------------------------------------------- |
| **Web 框架** | FastAPI + Uvicorn                                             |
| **工作流引擎** | LangGraph (有向图状态机编排)                                     |
| **向量数据库** | Milvus 3.0 (稠密+稀疏混合检索)                                   |
| **图数据库** | Neo4j 5.15 (知识图谱存储与检索)                                  |
| **文档数据库** | MongoDB 6.0 (对话历史持久化)                                      |
| **对象存储** | MinIO (文件与图片存储)                                            |
| **嵌入模型** | BGE-M3 (1024维稠密 + 稀疏向量，本地部署)                        |
| **重排序模型** | BGE-Reranker-Large (Cross-Encoder 精排，本地部署)                |
| **大语言模型** | 通义千问 Qwen-Flash (百炼 API，兼容 OpenAI 格式)                   |
| **PDF 解析** | MinerU API (PDF → Markdown)                                      |
| **联网搜索** | 百炼 MCP WebSearch (Streamable HTTP)                             |
| **RAG 评估** | RAGAS 0.2 (Faithfulness / ContextRecall / FactualCorrectness)    |
| **依赖管理** | uv + pyproject.toml                                              |
| **容器化** | Docker Compose (MongoDB + Neo4j)                                  |

---

## 📁 项目结构

```
dataset_rag/
├── app/
│   ├── clients/              # 外部服务客户端
│   │   ├── milvus_utils.py       # Milvus 向量库 (混合搜索/批量查询)
│   │   ├── neo4j_utils.py        # Neo4j 图库 (三元组写入/事实检索)
│   │   ├── mongo_history_utils.py # MongoDB 对话历史
│   │   └── minio_utils.py        # MinIO 对象存储
│   ├── conf/                 # 配置模块 (dataclass + .env)
│   │   ├── milvus_config.py
│   │   ├── embedding_config.py
│   │   ├── lm_config.py
│   │   ├── reranker_config.py
│   │   ├── ragas_config.py
│   │   └── ...
│   ├── core/                 # 核心工具
│   │   ├── load_prompt.py        # 提示词加载与变量渲染
│   │   └── logger.py             # Loguru 统一日志
│   ├── lm/                   # 模型工具
│   │   ├── lm_utils.py           # LLM 客户端 (ChatOpenAI 兼容)
│   │   ├── embedding_utils.py    # BGE-M3 向量生成
│   │   └── reranker_utils.py     # BGE-Reranker 重排序
│   ├── import_process/       # 📥 文档导入模块
│   │   ├── api/import_server.py  # FastAPI 导入服务 (端口 8000)
│   │   ├── agent/
│   │   │   ├── main_graph.py     # LangGraph 导入工作流
│   │   │   ├── state.py          # 导入状态定义
│   │   │   └── nodes/            # 导入节点
│   │   │       ├── node_entry.py               # 入口节点
│   │   │       ├── node_pdf_to_md.py            # PDF → Markdown
│   │   │       ├── node_md_img.py               # 图片处理
│   │   │       ├── node_document_split.py       # 文档分块
│   │   │       ├── node_item_name_recognition.py # 产品名识别
│   │   │       ├── node_bge_embedding.py        # BGE-M3 向量化
│   │   │       ├── node_import_milvus.py        # Milvus 入库
│   │   │       └── node_kg_extract_import.py    # KG 三元组抽取
│   │   └── page/import.html     # 导入管理页面
│   ├── query_process/        # 🔍 查询模块
│   │   ├── api/query_server.py   # FastAPI 查询服务 (端口 8001)
│   │   ├── agent/
│   │   │   ├── main_graph.py     # LangGraph 查询工作流
│   │   │   ├── state.py          # 查询状态定义
│   │   │   └── nodes/            # 查询节点
│   │   │       ├── node_item_name_confirm.py     # 产品名确认 + 问题重写
│   │   │       ├── node_search_embedding.py      # 向量检索
│   │   │       ├── node_search_embedding_hyde.py # HyDE 检索
│   │   │       ├── node_search_kg.py             # 知识图谱检索
│   │   │       ├── node_web_search_mcp.py        # 联网搜索 (MCP)
│   │   │       ├── node_rrf.py                   # RRF 融合排序
│   │   │       ├── node_rerank.py                # Reranker 精排
│   │   │       └── node_answer_output.py         # 答案生成 + SSE 输出
│   │   ├── sse/                 # SSE 工具
│   │   └── page/                # 前端页面
│   │       ├── chat.html            # 对话页面
│   │       └── query_monitor.html   # 查询监控页面
│   ├── eval/                  # 📊 RAGAS 评估模块
│   │   ├── ragas_evaluator.py
│   │   └── bge_m3_lc_embeddings.py
│   ├── utils/                 # 通用工具
│   │   ├── sse_utils.py           # SSE 事件推送
│   │   ├── task_utils.py          # 任务状态管理
│   │   ├── format_utils.py
│   │   └── ...
│   └── tool/                  # 模型下载脚本
├── prompts/                   # 提示词模板
│   ├── rewritten_query_and_itemnames.prompt  # 问题重写 + 产品名提取
│   ├── answer_out.prompt                     # 最终答案生成
│   ├── hyde_prompt.prompt                    # HyDE 假设性答案
│   ├── kg_extract.prompt                     # 知识图谱三元组抽取
│   ├── item_name_recognition.prompt          # 产品名识别
│   └── ...
├── doc/                       # 测试文档 (PDF)
├── output/                    # 输出目录 (解析结果)
├── docker-compose.yml         # Docker 编排 (MongoDB + Neo4j)
├── pyproject.toml             # 项目配置与依赖
└── .env                       # 环境变量配置
```

---

## 🚀 快速开始

### 环境要求

- **Python** ≥ 3.13
- **Docker** & **Docker Compose** (或自行安装 MongoDB + Neo4j)
- **Milvus** 向量数据库 (需单独部署或连接已有实例)
- **NVIDIA GPU** (可选，向量化与重排序支持 CUDA 加速)

### 1. 克隆项目

```bash
git clone https://github.com/your-username/dataset_rag.git
cd dataset_rag
```

### 2. 配置环境变量

复制并编辑 `.env` 文件，根据你的实际环境修改配置：

```bash
# 必填：LLM API 配置
OPENAI_API_KEY=sk-your-bailian-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-flash

# 必填：Milvus 向量数据库地址
MILVUS_URL=http://your-milvus-host:19530

# 必填：MongoDB (历史对话)
MONGO_URL=mongodb://your-mongo-host:27017
MONGO_DB_NAME=kb002

# 可选：Neo4j (知识图谱，不配则自动跳过)
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# 必填：BGE-M3 模型路径
BGE_M3_PATH=/path/to/bge-m3
BGE_DEVICE=cpu           # 或 cuda:0

# 必填：BGE-Reranker 模型路径
BGE_RERANKER_LARGE=/path/to/bge-reranker-large
BGE_RERANKER_DEVICE=cpu  # 或 cuda:0

# 必填：MinerU PDF 解析 API
MINERU_API_TOKEN=your-mineru-token
MINERU_BASE_URL=https://mineru.net/api/v4

# 可选：百炼 MCP 联网搜索
MCP_DASHSCOPE_BASE_URL_STREAMABLE=https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp

# 可选：MinIO 对象存储
MINIO_ENDPOINT=your-minio-host:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### 3. 启动依赖服务 (Docker)

```bash
# 启动 MongoDB (历史对话) + Neo4j (知识图谱)
docker-compose up -d

# 验证服务状态
docker-compose ps
```

> **注意**：Milvus 向量数据库需要单独部署。推荐使用 [Milvus Standalone](https://milvus.io/docs/install_standalone-docker.md) Docker 部署方式。

### 4. 安装 Python 依赖

推荐使用 `uv` 管理依赖：

```bash
# 安装 uv (如未安装)
pip install uv

# 安装项目依赖
uv sync
```

或使用传统 pip：

```bash
pip install -e .
```

### 5. 下载模型

运行脚本下载 BGE-M3 和 BGE-Reranker 模型到本地：

```bash
# 下载 BGE-M3 嵌入模型
python -m app.tool.download_bgem3

# 下载 BGE-Reranker 重排序模型
python -m app.tool.download_reranker
```

或通过 ModelScope 手动下载后，在 `.env` 中指定本地路径。

### 6. 启动服务

分别启动导入服务和查询服务：

```bash
# 终端 1：启动文档导入服务 (端口 8000)
python -m app.import_process.api.import_server

# 终端 2：启动查询服务 (端口 8001)
python -m app.query_process.api.query_server
```

启动后访问：
- 导入管理页面：http://localhost:8000/import
- 对话页面：http://localhost:8001/chat.html
- 监控页面：http://localhost:8001/query_monitor.html
- API 文档 (Swagger)：http://localhost:8000/docs │ http://localhost:8001/docs

---

## 📡 API 接口

### 文档导入服务 (端口 8000)

| 方法 | 路径             | 说明                        |
| ---- | ---------------- | --------------------------- |
| GET  | `/import`        | 导入管理页面 (HTML)          |
| POST | `/upload`        | 上传 PDF/MD 文件，异步导入    |
| GET  | `/status/{task_id}` | 查询导入任务进度与状态     |

**上传文件示例：**

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@hak180产品安全手册.pdf"
```

**响应：**

```json
{
  "code": 200,
  "message": "完成了文件上传，并开启了异步任务！文件数量为: 1",
  "task_ids": ["79b183aa-907f-491a-a339-4bfbae62db55"]
}
```

### 查询服务 (端口 8001)

| 方法   | 路径                  | 说明                        |
| ------ | --------------------- | --------------------------- |
| GET    | `/health`             | 健康检查                    |
| GET    | `/chat.html`          | 对话页面 (HTML)             |
| POST   | `/query`              | 发起提问 (支持同步/流式)     |
| GET    | `/stream/{session_id}` | SSE 长连接，接收流式结果     |
| GET    | `/history/{session_id}` | 查询历史对话                |
| DELETE | `/history/{session_id}` | 清空历史对话                |

**同步提问示例：**

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "HAK 180 烫金机怎么操作？",
    "is_stream": false
  }'
```

**流式提问示例：**

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "HAK 180 烫金机怎么操作？",
    "session_id": "my-session-001",
    "is_stream": true
  }'

# 在另一个终端连接 SSE
curl -N http://localhost:8001/stream/my-session-001
```

---

## 📊 RAG 评估

项目内置基于 RAGAS 的 RAG 评估模块，支持 4 项核心指标：

| 指标                   | 说明                                      |
| ---------------------- | ----------------------------------------- |
| **Faithfulness**       | 生成答案对检索上下文的忠实度              |
| **LLMContextRecall**   | 检索上下文对标准答案的覆盖率              |
| **FactualCorrectness** | 答案的事实正确性                          |
| **ResponseRelevancy**  | 答案与问题的相关度 (可选开启)             |

**运行评估：**

```bash
# 全量评估
python -m app.eval.ragas_evaluator

# 快速测试 (仅前 5 条)
python -m app.eval.ragas_evaluator --limit 5
```

评估数据集格式 (`JSON`):

```json
[
  {
    "question": "HAK 180 烫金机如何设置温度？",
    "ground_truth": "开机后按温度设置键，输入目标温度，默认建议 110°C",
    "item_name": "HAK 180 烫金机"
  }
]
```

---

## 🧠 知识图谱

系统支持从文档中自动抽取知识图谱三元组，存储在 Neo4j 中。支持的关系类型：

| 关系类型        | 说明               | 示例                                   |
| --------------- | ------------------ | -------------------------------------- |
| `HAS_PART`      | 组成部件           | HAK180 → HAS_PART → 烫金板             |
| `HAS_SPEC`      | 规格参数           | HAK180 → HAS_SPEC → 工作温度 110°C    |
| `HAS_OPERATION` | 操作步骤           | HAK180 → HAS_OPERATION → 开机步骤      |
| `HAS_WARNING`   | 安全警告           | 烫金板 → HAS_WARNING → 高温烫伤风险     |
| `HAS_RISK`      | 风险提示           | 误操作 → HAS_RISK → 设备损坏           |
| `HAS_FAULT`     | 常见故障           | 无法加热 → HAS_FAULT → 温度不上升       |
| `POSSIBLE_CAUSE` | 故障原因           | 无法加热 → POSSIBLE_CAUSE → 保险丝熔断 |
| `SOLUTION`      | 解决方案           | 保险丝熔断 → SOLUTION → 更换保险丝     |
| `REQUIRES`      | 前置条件           | 烫金操作 → REQUIRES → 预热完成         |
| `RELATED_TO`    | 通用关联 (兜底)    | -                                      |

知识图谱节点标签：
- **KGItem** — 产品/文档顶层实体
- **KGEntity** — 概念/操作/参数等具体实体
- **KGChunk** — 三元组证据来源的文档切片
- **EVIDENCED_BY** — 实体与证据切片的关联

在 Neo4j Browser (http://localhost:7474) 中可可视化浏览知识图谱。

---

## ⚙️ 配置说明

所有配置通过 `.env` 文件 + `dataclass` 配置类管理，主要配置项：

| 变量名                 | 说明               | 默认值                              |
| ---------------------- | ------------------ | ----------------------------------- |
| `OPENAI_API_KEY`       | LLM API 密钥       | -                                   |
| `OPENAI_BASE_URL`      | LLM API 地址       | 百炼兼容模式                        |
| `LLM_DEFAULT_MODEL`    | 默认大模型         | qwen-flash                          |
| `MILVUS_URL`           | Milvus 向量库地址  | -                                   |
| `MONGO_URL`            | MongoDB 地址       | -                                   |
| `NEO4J_URI`            | Neo4j 地址         | bolt://localhost:7687               |
| `BGE_M3_PATH`          | BGE-M3 模型路径    | -                                   |
| `BGE_DEVICE`           | 向量化设备         | cpu                                 |
| `BGE_RERANKER_LARGE`   | Reranker 模型路径  | -                                   |
| `MINERU_API_TOKEN`     | MinerU API 密钥    | -                                   |
| `LOG_CONSOLE_LEVEL`    | 控制台日志级别     | INFO                                |
| `LOG_FILE_RETENTION`   | 日志保留时间       | 7 days                              |

---

## 📋 开发计划

- [x] PDF/Markdown 文档导入与向量化
- [x] 多路召回 (向量 + HyDE + 联网搜索 + 知识图谱)
- [x] RRF 融合 + Reranker 精排
- [x] SSE 流式输出
- [x] 知识图谱三元组抽取与检索
- [x] RAGAS 评估体系
- [x] 多轮对话记忆
- [ ] Docker 一键部署 (含 Milvus)
- [ ] 更多文档格式支持 (Word/Excel)
- [ ] 前端对话界面优化
- [ ] 用户权限与多租户管理
- [ ] 模型热切换与 A/B 测试
- [ ] 知识图谱可视化编辑

---

## 📄 License

MIT License

---

## 🙏 致谢

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — BAAI 多语言嵌入模型
- [BGE-Reranker](https://huggingface.co/BAAI/bge-reranker-large) — BAAI 重排序模型
- [Milvus](https://milvus.io/) — 开源向量数据库
- [LangGraph](https://www.langchain.com/langgraph) — LangChain 图状态机
- [RAGAS](https://docs.ragas.io/) — RAG 评估框架
- [MinerU](https://github.com/opendatalab/MinerU) — PDF 解析工具
- [通义千问](https://tongyi.aliyun.com/) — 阿里云百炼大模型

---

> ⭐ 如果这个项目对你有帮助，欢迎 Star & Fork！
