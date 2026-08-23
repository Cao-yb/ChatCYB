import streamlit as st
import uuid
from openai import AuthenticationError
from backend import send_messages_to_LLM

# 侧边栏：用户输入 API 密钥（阿里云 DashScope）
with st.sidebar:

    api_key = (st.text_input("请输入您的阿里云(DashScope) API 密钥:" , type="password") or "").strip()
    #超链接提示用户获取阿里云 API 密钥, 点击跳转阿里云百炼控制台
    st.markdown("[获取阿里云 DashScope API密钥](https://bailian.console.aliyun.com/?apiKey=1)")
    aliyun_API_key = api_key

st.title("ChatCYB")
st.write("（基于阿里云 DashScope 开发, 支持文字、文件问答、数据分析、联网搜索）")

# 用 session_state 保存聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 每个浏览器会话分配一个独立的记忆编号：
# 同一会话内连续对话有记忆；刷新页面后 session_state 清空会生成新编号，
# 新对话即“失忆”，也避免不同用户之间的记忆串号
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框：既支持输入文字，也支持上传文件（accept_file=True）
# docx/pdf/txt 等 → 后端解析成文本给 AI 阅读；csv/excel/json/parquet/tsv → 后端加载成表格供 AI 精确分析
if value := st.chat_input("请输入消息，回车发送", accept_file=True,
                          file_type=["txt", "md", "json", "csv", "tsv", "xlsx", "xls", "parquet",
                                     "py", "js", "html", "xml", "yaml", "log", "docx", "pdf"]):
    user_text = value.text or ""   # 用户输入的文字

    # 上传的文件原样（文件名 + 字节）交给后端：
    # 后端自动判断是"解析成文本阅读"还是"加载成 DataFrame 用 pandas 分析"
    files = [(f.name, f.getvalue()) for f in value.files] if value.files else []

    # 显示用户消息（文字 + 文件名）
    display = user_text
    if files:
        names = "、".join(name for name, _ in files)
        display += f"\n\n（上传文件：{names}）"
    st.chat_message("user").markdown(display)
    st.session_state.messages.append({"role": "user", "content": display})

    # 调用后端（文字 + 文件一起发送）
    if not api_key:
        st.warning("请输入您的阿里云(DashScope) API密钥")
    elif not (user_text.strip() or files):
        st.warning("消息为空，请输入文字或上传文件")
    else:
        with st.spinner("思考中..."):
            result = None
            try:
                result = send_messages_to_LLM(user_text, aliyun_API_key, st.session_state.thread_id, files=files)
            except AuthenticationError:
                # 401：密钥被阿里云拒绝（输错/拿成了 DeepSeek 的密钥/带了多余空格）
                st.error(
                    "API 密钥无效（401）：请检查侧边栏输入的是【阿里云 DashScope】的密钥，"
                    "而不是 DeepSeek 或其他平台的密钥；重新复制时注意不要带多余空格。"
                    "获取地址：https://bailian.console.aliyun.com/?apiKey=1"
                )
            except Exception as e:
                st.error(f"调用出错：{type(e).__name__}：{e}")
            if result is not None:
                st.chat_message("assistant").markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
