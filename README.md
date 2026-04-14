# CodeReview Agent

基于 LangGraph 的多 Agent 并行代码审查系统。

## 功能
- Supervisor Agent 拆解审查任务
- Style / Logic / Security 三个专家 Agent 并行审查
- Self-Critique 评分不达标自动触发修复
- LLM 自动修复代码
- Streamlit 可视化界面

## 技术栈
- LangGraph
- LangChain
- 通义千问 qwen3-max
- Streamlit

## 使用方法
1. 安装依赖：pip install -r requirements.txt
2. 配置环境变量：复制 .env.example 为 .env，填入 API Key
3. 运行：streamlit run app.py