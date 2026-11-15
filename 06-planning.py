"""
06. Planning 패턴 (Plan-and-Execute)

복잡한 작업을 바로 수행하지 않고, 먼저 "어떤 단계로 처리할지 계획"을 세운 뒤
계획의 각 단계를 하나씩 실행하고, 마지막에 결과를 종합하는 패턴.

흐름:
    주제 입력
      → [Planner]     계획 수립: 작성할 항목을 글머리 기호 목록으로 생성
      → [Executor]    계획 실행 루프: 단계 하나씩 LLM으로 작성 → 결과 누적 → 남은 단계 있으면 반복
      → [Synthesizer] 종합: 단계별 결과를 모아 최종 요약문 작성
      → 최종 결과 출력

왜 쓰는가:
    - "계획"이 명시적인 상태(state)로 존재하므로 진행 상황을 추적하고, 단계별로 검증/재시도할 수 있다.
    - 한 번의 거대한 프롬프트보다 각 단계의 프롬프트가 단순해져 결과가 안정적이다.

LangGraph 사용 포인트:
    - StateGraph: 노드(함수) 간에 공유되는 상태(TypedDict)를 두고 그래프로 흐름을 정의한다.
    - 조건부 엣지(add_conditional_edges): "남은 단계가 있는가?"에 따라 루프를 돌지 종료할지 결정한다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
Fake 모델은 "계획 → 1단계 → 2단계 → 3단계 → 종합" 순서로 총 5개의 응답을 돌려준다.
"""
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. Fake 모델 정의
#    그래프가 LLM을 호출하는 순서(계획 1회 → 단계 실행 3회 → 종합 1회)에 맞춰 응답을 배치한다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # [Planner] 계획: 글머리 기호 3개 → 3단계로 파싱된다
    """- 강화학습의 기본 개념 (에이전트, 환경, 보상)
- 강화학습이 AI 발전에 기여한 대표 사례
- 강화학습의 한계와 앞으로의 과제""",

    # [Executor] 1단계 결과
    "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 스스로 학습하는 방법이다. "
    "정답 레이블 없이 시행착오를 통해 전략(정책)을 개선한다는 점에서 지도학습과 구별된다.",

    # [Executor] 2단계 결과
    "알파고의 바둑 정복, 로봇 제어, 추천 시스템 최적화, 그리고 대규모 언어 모델의 인간 피드백 기반 학습(RLHF)까지 "
    "강화학습은 AI가 복잡한 의사결정 문제를 해결하는 핵심 도구로 자리잡았다.",

    # [Executor] 3단계 결과
    "다만 많은 상호작용 데이터가 필요하고, 보상 함수 설계가 어려우며, 학습이 불안정할 수 있다. "
    "샘플 효율성 개선과 안전한 탐색이 앞으로의 주요 과제다.",

    # [Synthesizer] 종합 요약
    """강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하도록 스스로 학습하는 AI 방법론이다.
정답 없이 시행착오로 정책을 개선한다는 점에서 지도학습과 다르며, 알파고·로봇 제어·추천 시스템·RLHF 등
복잡한 의사결정 문제에서 핵심 역할을 해왔다. 그러나 대량의 상호작용 데이터, 까다로운 보상 설계, 학습 불안정성이라는
한계가 있어 샘플 효율성과 안전한 탐색이 향후 과제로 남아 있다. 이러한 과제에도 불구하고 강화학습은
AI가 스스로 전략을 익히게 하는 가장 유력한 접근법으로, 그 중요성은 앞으로 더욱 커질 것이다."""
])


# ---------------------------------------------------------------------------
# 2. 그래프 상태 정의
#    모든 노드가 이 딕셔너리를 읽고, 변경할 키만 반환하면 LangGraph가 병합해 준다.
# ---------------------------------------------------------------------------
class PlanExecuteState(TypedDict):
    topic: str              # 사용자가 준 주제
    plan: list[str]         # Planner가 만든 단계 목록
    current_step: int       # 지금 실행할 단계의 인덱스
    step_results: list[str] # 단계별 실행 결과 (누적)
    final_summary: str      # Synthesizer가 만든 최종 요약


# ---------------------------------------------------------------------------
# 3. 노드 정의 (각 노드는 state를 받아 "변경할 키"만 담은 dict를 반환한다)
# ---------------------------------------------------------------------------
def planner_node(state: PlanExecuteState) -> dict:
    """[Planner] 주제를 보고 작성할 항목(단계)을 글머리 기호 목록으로 계획한다."""
    print("\n>>> [Planner] 계획 수립 중...")

    response = llm.invoke([
        SystemMessage(content="당신은 기술 문서 작성 전문가이자 콘텐츠 전략가입니다. "
                              "글을 쓰기 전에 명확하고 실행 가능한 계획을 먼저 세우는 것이 강점입니다."),
        HumanMessage(content=f"주제 '{state['topic']}'에 대한 요약문을 작성하려고 합니다.\n"
                             f"다뤄야 할 핵심 항목을 3개의 글머리 기호(- )로 나열하세요. 항목만 출력하세요."),
    ])

    # "- 항목" 형식의 줄만 골라 계획 리스트로 변환한다.
    plan = [line.lstrip("- ").strip() for line in response.content.splitlines() if line.strip().startswith("-")]

    print("--- 수립된 계획 ---")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    return {"plan": plan, "current_step": 0, "step_results": []}


def executor_node(state: PlanExecuteState) -> dict:
    """[Executor] 계획의 현재 단계 하나를 LLM으로 작성하고 결과를 누적한다."""
    step_index = state["current_step"]
    step = state["plan"][step_index]
    print(f"\n>>> [Executor] {step_index + 1}/{len(state['plan'])} 단계 실행: '{step}'")

    response = llm.invoke([
        SystemMessage(content="당신은 기술 문서 작성 전문가입니다. 주어진 항목에 대해 2~3문장으로 간결하게 작성하세요."),
        HumanMessage(content=f"주제: {state['topic']}\n작성할 항목: {step}"),
    ])

    print(f"--- 단계 결과 ---\n{response.content}")

    # 기존 결과 리스트에 이번 결과를 덧붙이고, 다음 단계 인덱스로 넘긴다.
    return {
        "step_results": state["step_results"] + [response.content],
        "current_step": step_index + 1,
    }


def synthesizer_node(state: PlanExecuteState) -> dict:
    """[Synthesizer] 단계별 결과를 하나의 최종 요약문으로 종합한다."""
    print("\n>>> [Synthesizer] 최종 요약 작성 중...")

    # 계획 항목과 실행 결과를 짝지어 LLM에 전달한다.
    sections = "\n\n".join(
        f"### {step}\n{result}" for step, result in zip(state["plan"], state["step_results"])
    )

    response = llm.invoke([
        SystemMessage(content="당신은 기술 문서 작성 전문가입니다. 아래 섹션들을 바탕으로 "
                              "약 200자 내외의 읽기 쉬운 최종 요약문을 작성하세요."),
        HumanMessage(content=f"주제: {state['topic']}\n\n{sections}"),
    ])

    return {"final_summary": response.content}


# ---------------------------------------------------------------------------
# 4. 조건부 분기 함수
#    executor 실행 후 "남은 단계가 있으면 executor로 되돌아가고, 없으면 synthesizer로" 진행한다.
#    반환값은 add_conditional_edges에 등록한 매핑의 키와 일치해야 한다.
# ---------------------------------------------------------------------------
def should_continue(state: PlanExecuteState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "continue"   # 남은 단계 있음 → 루프
    return "finish"         # 모든 단계 완료 → 종합


# ---------------------------------------------------------------------------
# 5. 그래프 조립
#    START → planner → executor ⇄ (조건부) → synthesizer → END
# ---------------------------------------------------------------------------
graph_builder = StateGraph(PlanExecuteState)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("executor", executor_node)
graph_builder.add_node("synthesizer", synthesizer_node)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue": "executor",     # 루프백
        "finish": "synthesizer",
    },
)
graph_builder.add_edge("synthesizer", END)

# compile()하면 invoke/stream이 가능한 실행 가능 그래프가 된다.
planning_agent = graph_builder.compile()


# ---------------------------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------------------------
def main() -> None:
    topic = "AI에서 강화학습의 중요성"
    print(f"## 계획 수립 및 작성 작업 실행: '{topic}' ##")

    # 초기 상태에는 topic만 주면 된다. 나머지 키는 노드들이 채운다.
    result = planning_agent.invoke({"topic": topic})

    print("\n\n" + "=" * 30 + " 최종 결과 " + "=" * 30)
    print("\n### 계획")
    for i, step in enumerate(result["plan"], 1):
        print(f"- {i}. {step}")
    print("\n### 요약")
    print(result["final_summary"])


if __name__ == "__main__":
    main()
