import streamlit as st
import uuid
from backend import send_messages_to_LLM, extract_text

# 侧边栏：用户输入 API 密钥
with st.sidebar:

    api_key = st.text_input("请输入您的DeepSeek API 密钥:" , type="password")
    #超链接提示用户获取DeepSeek API, 点击跳转DeepSeek官网
    st.markdown("[获取DeepSeek API密钥](https://platform.deepseek.com)")
    deepseek_API_key = api_key

st.title("ChatCYB")
st.write("（基于DeepSeek 开发, 支持文字、部分文件处理）")

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

# 输入框：既支持输入文字，也支持上传 DeepSeek 能理解的文件（accept_file=True）
if value := st.chat_input("请输入消息，回车发送", accept_file=True,
                          file_type=["txt", "md", "json", "csv", "py", "js", "html",
                                     "xml", "yaml", "log", "docx"]):
    user_text = value.text or ""   # 用户输入的文字

    # 解析上传的文件为文本
    file_text = ""
    for f in value.files:
        content = extract_text(f.name, f.getvalue())
        if content:
            if len(content) > 20000:  # 防止内容过长超出模型上下文窗口
                content = content[:20000] + "\n...[内容过长，已截断]"
            file_text += f"\n--- 文件[{f.name}]的内容 ---\n{content}\n"
        else:
            st.warning(f"无法解析文件 {f.name}，已跳过")

    # 显示用户消息（文字 + 文件名）
    display = user_text
    if value.files:
        names = "、".join(f.name for f in value.files)
        display += f"\n\n（上传文件：{names}）"
    st.chat_message("user").markdown(display)
    st.session_state.messages.append({"role": "user", "content": display})

    # 调用后端（文字 + 文件内容合并后发送）
    if not api_key:
        st.warning("请输入您的DeepSeek API密钥")
    elif not (user_text.strip() or file_text.strip()):
        st.warning("消息为空，请输入文字或上传文件")
    else:
        with st.spinner("思考中..."):
            result = send_messages_to_LLM(user_text + file_text, deepseek_API_key, st.session_state.thread_id)
        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
