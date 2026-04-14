import json
import operator
import re
from typing import TypedDict, Annotated, Literal


class ReviewState(TypedDict):
    code: str  # 一开始传入的
    language: str  # 传入的代码语言
    subtasks: Annotated[list[dict], operator.add]  # supervisor 填写
    worker_results: Annotated[list[dict], operator.add]  # 三个worker填写
    final_report: str  # aggregator填写
    review_score: int  # aggregator填写
    critique_count: int  # 记录重审次数
    critique_feedback: str  # 记录批评意见，下次审查时参考
    fixed_code: str  # 存修复后的代码


class WorkerState(TypedDict):
    task_id: str  # 我是哪个专家
    focus: str  # 我的审查重点
    code: str  # 要审查的代码
    language: str
    result: str  # 我的审查结果


from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatTongyi(model="qwen3-max")

SUPERVISOR_PROMPT = """你是代码审查主管。
将审查任务拆解为三个子任务，以 JSON 列表返回：
[
  {"task_id": "style_check",    "focus": "代码风格检查"},
  {"task_id": "logic_check",    "focus": "逻辑和Bug检查"},
  {"task_id": "security_check", "focus": "安全漏洞检查"}
]
只返回 JSON，不要其他内容。"""

STYLE_PROMPT = """你是代码风格专家，检查：
- 命名规范、注释质量、代码格式
输出：## 风格问题\n- [高/中/低] 问题描述"""

LOGIC_PROMPT = """你是代码逻辑专家，检查：
- 边界条件、异常处理、潜在Bug
输出：## 逻辑问题\n- [高/中/低] 问题描述"""

SECURITY_PROMPT = """你是代码安全专家，检查：
- SQL注入、硬编码密码、输入验证
输出：## 安全问题\n- [高/中/低] 问题描述"""

AGGREGATE_PROMPT = """将三个专家的审查结果整合成报告。
格式：
# 代码审查报告
## 总体评分（0-10分）
## 问题汇总
## 优先修复建议

最后一行必须是：SCORE: X"""

CRITIQUE_PROMPT = """你是代码审查质量评估专家。
给定一份代码审查报告，判断它是否足够全面和准确。

如果报告有以下问题，请指出：
- 遗漏了重要的安全漏洞
- 没有给出具体的修复建议
- 问题描述不够清晰

输出格式：
## 不足之处
- 具体问题描述

## 改进要求
下一轮审查需要重点关注的内容"""

FIX_PROMPT = """你是资深代码修复专家。
根据代码审查报告中指出的问题，对代码进行修复。

修复要求：
- 修复所有高严重度问题
- 修复安全漏洞（SQL注入、硬编码密码等）
- 修复逻辑错误（除零、空指针等）
- 保持代码原有功能不变

只返回修复后的完整代码，不要任何解释。"""


def supervisor_decompose(state: ReviewState):
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f'代码：\n {state["code"]}')
    ]
    response = llm.invoke(messages)
    content = response.content.strip()  # .strip() 去掉首尾的空格和换行。为什么要 strip？因为 LLM 有时候会在回答前后加空行，不去掉的话 JSON 解析会失败。

    try:
        if "```" in content:
            content = re.search(r'\[[\s\S]*]', content).group()
        subtasks = json.loads(content)
    except Exception as e:
        print(f"JSON 解析失败：{e}")
        subtasks = [
            {"task_id": "style_check", "focus": "代码风格检查"},
            {"task_id": "logic_check", "focus": "逻辑检查"},
            {"task_id": "security_check", "focus": "安全检查"},
        ]
    return {"subtasks": subtasks}


from langgraph.types import Send


def dispatch(state: ReviewState) -> list[Send]:
    return [
        Send("worker_invoker", {
            "task_id": subtask["task_id"],
            "focus": subtask["focus"],
            "code": state["code"],
            "language": state["language"]
        })
        for subtask in state["subtasks"]
    ]


def worker_invoker(state: WorkerState) -> dict:
    prompt_map = {
        "style_check": STYLE_PROMPT,
        "logic_check": LOGIC_PROMPT,
        "security_check": SECURITY_PROMPT
    }

    system_prompt = prompt_map.get(state["task_id"], LOGIC_PROMPT)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(f'语言：{state["language"]}\n\n'
                              f'代码：\n {state["code"]}'))
    ]

    response = llm.invoke(messages)

    return {
        "worker_results": [{
            "task_id": state["task_id"],
            "focus": state["focus"],
            "result": response.content,
        }]
    }


def aggregate(state: ReviewState) -> dict:
    # 第一步：把三个 worker 的结果拼成一段上下文
    context = "\n\n".join(
        f"### {r['task_id']}\n{r['result']}"
        for r in state["worker_results"]
    )

    # 第二步：调用 LLM 生成最终报告
    messages = [
        SystemMessage(content=AGGREGATE_PROMPT),
        HumanMessage(content=(
            f"代码：\n{state['code']}\n\n"
            f"三个专家的结果：\n{context}"
        )),
    ]
    response = llm.invoke(messages)
    content = response.content

    # 第三步：用正则从报告里提取评分
    score = 5  # 默认分数，解析失败时用
    match = re.search(r'SCORE:\s*(\d+)', content)
    if match:
        score = min(10, max(0, int(match.group(1))))

    return {
        "final_report": content,
        "review_score": score,
    }


def critique(state: ReviewState) -> dict:
    # 达标或者已经重审2次，不再继续
    if state["review_score"] >= 6 or state["critique_count"] >= 2:
        return {}  # 返回空字典，state 不变，直接往后走

    # 不达标，让 LLM 生成改进意见
    messages = [
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=(
            f"当前评分：{state['review_score']}/10\n\n"
            f"审查报告：\n{state['final_report']}"
        )),
    ]
    response = llm.invoke(messages)

    return {
        "critique_feedback": response.content,
        "critique_count":    state["critique_count"] + 1,
        # 重置 worker_results 和 subtasks，下一轮重新收集
        "worker_results":    [],
        "subtasks":          [],
    }


def fix_code(state: ReviewState) -> dict:
    messages = [
        SystemMessage(content=FIX_PROMPT),
        HumanMessage(content=(
            f"编程语言：{state['language']}\n\n"
            f"原始代码：\n```{state['language']}\n{state['code']}\n```\n\n"
            f"审查报告：\n{state['final_report']}\n\n"
            f"改进要求：\n{state['critique_feedback']}"
        )),
    ]
    response = llm.invoke(messages)

    # 提取代码块里的内容，去掉 LLM 可能加的 markdown 格式
    fixed = response.content
    match = re.search(r'```[\w]*\n([\s\S]*?)```', fixed)
    if match:
        fixed = match.group(1)

    print(f"\n第 {state['critique_count']} 轮修复完成，重新审查...")

    return {
        "code":           fixed,  # 用修复后的代码替换原来的，下一轮审查新代码
        "fixed_code":     fixed,
        "worker_results": [],     # 清空，重新审查
        "subtasks":       [],
    }


def should_retry(state: ReviewState) -> Literal["fix_code", "__end__"]:
    # 不达标 且 重审次数未超限 → 去修复代码
    if state["review_score"] < 6 and state["critique_count"] < 2:
        print(f"评分 {state['review_score']} 不达标，触发第 {state['critique_count']} 轮自动修复...")
        return "fix_code"
    # 达标 或 已达上限 → 结束
    print(f"评分 {state['review_score']}，审查完成")
    return "__end__"


from langgraph.graph import StateGraph, START, END


def build_graph():
    # 第一步：创建图，告诉它用 ReviewState 作为状态
    graph = StateGraph(ReviewState)

    # 第二步：注册所有节点
    graph.add_node("supervisor_decompose", supervisor_decompose)
    graph.add_node("worker_invoker",       worker_invoker)
    graph.add_node("aggregate",            aggregate)
    graph.add_node("critique",             critique)   # 质量评估节点
    graph.add_node("fix_code",             fix_code)   # 自动修复节点

    # 第三步：连接边
    graph.add_edge(START, "supervisor_decompose")

    graph.add_conditional_edges(
        "supervisor_decompose",  # 从这个节点出发
        dispatch,                # 调用 dispatch 决定去哪里
        ["worker_invoker"]       # 可能的目标节点列表
    )

    graph.add_edge("worker_invoker", "aggregate")
    graph.add_edge("aggregate",      "critique")       # aggregate 后接 critique

    # critique 后根据评分决定修复还是结束
    graph.add_conditional_edges(
        "critique",
        should_retry,
        ["fix_code", END]
    )

    # 修复完重新回到 supervisor 重新审查
    graph.add_edge("fix_code", "supervisor_decompose")

    # 第四步：编译
    return graph.compile()


# 测试代码
if __name__ == "__main__":
    TEST_CODE = """
def get_user(user_id):
    password = "admin123"
    query = "SELECT * FROM users WHERE id = " + user_id
    return db.execute(query)

def divide(a, b):
    return a / b
"""

    compiled = build_graph()
    result = compiled.invoke({
        "code":              TEST_CODE,
        "language":          "python",
        "subtasks":          [],
        "worker_results":    [],
        "final_report":      "",
        "review_score":      0,
        "critique_count":    0,    # 初始重审次数为0
        "critique_feedback": "",   # 初始无改进意见
        "fixed_code":        "",   # 初始无修复代码
    })

    print("=" * 50)
    print("原始代码：")
    print(TEST_CODE)
    print("=" * 50)
    print("修复后代码：")
    print(result["fixed_code"] if result["fixed_code"] else "代码质量达标，无需修复")
    print("=" * 50)
    print("最终评分：", result["review_score"], "/ 10")
    print("=" * 50)
    print(result["final_report"])