
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import io
from docx import Document


# ---------- 解析上传文件为文本 ----------
def extract_text(filename, content):
    """根据文件扩展名把上传内容解析为文本，失败时返回空字符串"""
    try:
        name = filename.lower()
        if name.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            texts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    texts.append(" | ".join(cell.text for cell in row.cells))
            return "\n".join(texts)
        else:
            return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ---------- 全局唯一的记忆存储（在模块级创建，保证跨轮次对话记住上下文）----------
checkpointer = InMemorySaver()  # 自动记事本

# ---------- 发送消息到 Agent ----------
def send_messages_to_LLM(message, deepseek_API_key):
    model = ChatOpenAI(
    model="deepseek-v4-flash",
    openai_api_key=deepseek_API_key,
    openai_api_base="https://api.deepseek.com",
    )

    # ---------- 建一个带记忆的 Agent ----------
    agent = create_agent(model, checkpointer=checkpointer)

    # 固定写法：指定一个聊天窗口（必须写，不然会报错）
    config = {"configurable": {"thread_id": "default"}}
    messages = [
        SystemMessage(content="你是一个乐于助人的AI助手,基于DeepSeek-v4-flash开发,你具备记忆功能，当前所在的应用之中，你只能接收用户的文本输入，然后给用户返回输出，你无法接收并返回其他的信息"),
        HumanMessage(content=message),
    ]
    # LangChain 1.x 的 create_agent 需要传入字典（key 为 messages）
    response = agent.invoke({"messages": messages}, config=config)
    return response["messages"][-1].content