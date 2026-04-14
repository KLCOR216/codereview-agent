import streamlit as st
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Review_Correct_agent import build_graph


st.set_page_config(
    page_title="CodeReview Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 CodeReview Agent")
st.caption("基于 LangGraph 多 Agent 并行代码审查系统")


with st.sidebar:
    st.header("⚙️ 配置")
    language = st.selectbox(
        "编程语言",
        ["python", "javascript", "java", "go", "cpp"],
        index=0
    )
    max_retries = st.slider("最大重审次数", 0, 3, 2)
    st.divider()
    st.markdown("### 使用说明")
    st.markdown("""
    1. 在右侧输入代码或上传文件
    2. 点击开始审查
    3. 等待三个专家并行分析
    4. 查看审查报告和修复建议
    """)


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 输入代码")

    # 上传文件
    uploaded_file = st.file_uploader(
        "上传代码文件",
        type=["py", "js", "java", "go", "cpp", "ts", "txt"]
    )

    # 如果上传了文件，读取内容填入文本框
    if uploaded_file:
        code_content = uploaded_file.read().decode("utf-8")
    else:
        code_content = """def get_user(user_id):
    password = "admin123"
    query = "SELECT * FROM users WHERE id = " + user_id
    return db.execute(query)

def divide(a, b):
    return a / b"""

    # 代码输入框
    code_input = st.text_area(
        "或直接粘贴代码",
        value=code_content,
        height=350,
        placeholder="在这里粘贴你的代码..."
    )

    # 开始审查按钮
    start_btn = st.button("🚀 开始审查", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 审查结果")

    # 结果区域占位
    result_placeholder = st.empty()
    result_placeholder.info("等待代码输入...")


if start_btn:
    if not code_input.strip():
        st.error("请先输入代码！")
    else:
        with col2:
            result_placeholder.empty()

            # 显示进度
            with st.status("审查进行中...", expanded=True) as status:
                st.write("🔧 Supervisor 正在拆解任务...")
                st.write("🎨 Style Agent 检查代码风格...")
                st.write("🧠 Logic Agent 分析逻辑问题...")
                st.write("🔒 Security Agent 扫描安全漏洞...")

                try:
                    # 调用你的 Agent
                    compiled = build_graph()
                    result = compiled.invoke({
                        "code":              code_input,
                        "language":          language,
                        "subtasks":          [],
                        "worker_results":    [],
                        "final_report":      "",
                        "review_score":      0,
                        "critique_count":    0,
                        "critique_feedback": "",
                        "fixed_code":        "",
                    })
                    status.update(label="审查完成！", state="complete")
                except Exception as e:
                    status.update(label="审查失败", state="error")
                    st.error(f"出错了：{e}")
                    st.stop()

            # ---- 显示评分 ----
            score = result["review_score"]
            if score >= 8:
                st.success(f"✅ 最终评分：{score} / 10")
            elif score >= 6:
                st.warning(f"⚠️ 最终评分：{score} / 10")
            else:
                st.error(f"❌ 最终评分：{score} / 10")

            # ---- 显示审查报告 ----
            st.markdown("### 审查报告")
            st.markdown(result["final_report"])

            # ---- 显示修复后的代码 ----
            if result.get("fixed_code"):
                st.markdown("### 🔨 修复后的代码")
                st.code(result["fixed_code"], language=language)

                # 下载按钮
                st.download_button(
                    label="⬇️ 下载修复后的代码",
                    data=result["fixed_code"],
                    file_name=f"fixed_code.{language}",
                    mime="text/plain"
                )
            else:
                st.success("代码质量达标，无需修复")


st.divider()
st.subheader("💬 追问 Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("对审查结果有疑问？可以继续追问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 LLM 回答追问
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatTongyi(model="qwen3-max")
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = llm.invoke([
                SystemMessage(content="你是代码审查专家，回答用户关于代码审查的问题"),
                HumanMessage(content=prompt)
            ])
            st.markdown(response.content)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content
            })