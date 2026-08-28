# 学习Agent开发的学习项目
# ChatCYB — 智能对话助手

一个基于 LangChain Agent 构建的 Streamlit 聊天应用，支持**多轮对话记忆、联网搜索、文件问答、数据分析**四大能力。用户输入 API 密钥即可使用，无需任何配置文件。

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 💬 多轮对话记忆 | 基于 LangGraph `InMemorySaver`，同一会话内连续对话不忘上文 |
| 🌐 联网搜索 | 调用阿里云 DashScope 联网能力，可回答天气、新闻等实时问题 |
| 📄 文件问答 | 上传 docx / pdf / txt / 代码文件，AI 直接阅读并回答问题 |
| 📊 数据分析 | 上传 CSV / Excel / JSON / Parquet 等表格文件，AI 自动编写 pandas 代码精确计算 |
| 🔒 多用户隔离 | 每个浏览器会话独立 `thread_id`，对话记忆与数据缓存互不干扰 |

## 🏗️ 项目架构

```
frontend.py（Streamlit 界面）
   │  收集输入 / 展示对话 / 文件上传 / 错误提示
   ▼
backend.py（LangChain Agent）
   ├── 主 Agent（create_agent）
   │     ├── 🌐 联网搜索工具（DashScope web_search）
   │     └── 📊 数据分析工具
   │           └── 子 Agent + python_exec（LLM 写 pandas 代码并真实执行）
   ├── InMemorySaver（对话记忆，按 thread_id 隔离）
   └── 会话级数据缓存（上传一次，整场会话可反复分析）
```

### 工作流程

1. 前端收集文字与文件，连同 API 密钥、会话编号发给后端
2. 后端按文件类型分流：表格文件读成 DataFrame 进入分析工具；文档文件解析为文本进入提示词
3. 组装 Agent（模型 + 工具 + 系统提示 + 记忆），由模型自主决策是否调用工具
4. 数据类问题由 AI 生成 pandas 代码、真实执行后返回**精确计算结果**，杜绝模型心算误差

## 🛠️ 技术栈

- **界面**：Streamlit
- **Agent 框架**：LangChain 1.x（`create_agent` + Tool Calling）+ LangGraph
- **模型接入**：阿里云 DashScope（OpenAI 兼容接口），模型 `deepseek-v4-flash`
- **数据处理**：pandas
- **文件解析**：python-docx（Word）、pypdf（PDF）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/你的仓库.git
cd 你的仓库
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

```bash
streamlit run frontend.py
```

### 4. 使用

在侧边栏填入你的**阿里云 DashScope API 密钥**（[获取地址](https://bailian.console.aliyun.com/?apiKey=1)），即可开始对话。

## 💡 使用示例

- 直接提问：「今天洛阳市天气怎样？」→ 自动触发联网搜索
- 上传 CSV 后提问：「各月份销量分别是多少？哪个月最高？」→ 自动编写 pandas 代码精确计算
- 上传 PDF 后提问：「总结一下这份文档的核心观点」→ AI 阅读文档后作答

## ⚠️ 注意事项

- 数据分析功能会执行 AI 生成的 Python 代码，与官方 pandas agent 具有同等风险，**仅建议在本地可信环境使用**
- 文档类文件内容超过 20000 字符会被截断，建议上传前先行拆分
- 请妥善保管 API 密钥，本项目不会存储密钥（仅在前端输入、内存中传递）
