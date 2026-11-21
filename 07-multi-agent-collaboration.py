"""
07. Multi-Agent Collaboration 패턴

서로 다른 역할(페르소나)을 가진 여러 에이전트가 공유 상태(state)를 주고받으며
하나의 결과물을 함께 완성하는 패턴. 각 에이전트는 자기 전문 분야만 담당한다.

흐름:
    주제 입력
      → [Researcher] AI 트렌드 조사·요약             → research_findings
      → [Writer]     리서치 결과로 블로그 초안 작성   → draft
      → [Editor]     초안 검토
            ├─ 수정 요청 → 피드백을 담아 Writer에게 되돌림 (최대 N회)
            └─ 승인     → 최종 블로그 확정
      → 최종 블로그 출력

왜 쓰는가:
    - 하나의 프롬프트에 "조사하고, 쓰고, 검토하라"를 전부 넣는 것보다, 역할별로 프롬프트를 분리하면
      각 에이전트의 지시가 단순해지고 결과 품질이 안정적이다.
    - 에이전트 간 "핸드오프(handoff)"가 상태로 명시되므로 누가 무엇을 넘겼는지 추적할 수 있다.

LangGraph 사용 포인트:
    - 에이전트 하나 = 노드 하나. 역할은 시스템 프롬프트로 부여한다.
    - 에이전트 간 전달물은 공유 상태(TypedDict)의 키로 표현한다.
    - 편집자의 판정에 따라 조건부 엣지로 "재작성 루프" 또는 "종료"를 선택한다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
Fake 모델은 "리서치 → 초안 v1 → 편집자 수정 요청 → 초안 v2 → 편집자 승인" 순서로 5개의 응답을 돌려준다.
"""
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 입력 주제
TOPIC = "인공지능의 최신 트렌드"

MAX_REVISIONS = 2   # 편집자가 수정을 요청할 수 있는 최대 횟수 (무한 루프 방지)

# 에이전트 역할 정의
#    멀티 에이전트에서 "에이전트"는 결국 (역할을 부여하는 시스템 프롬프트 + 모델) 조합이다.
#    역할마다 시스템 프롬프트를 분리해 두면, 각 노드에서 자기 역할로만 LLM을 호출할 수 있다.
RESEARCHER_ROLE = (
    "당신은 시니어 리서치 분석가입니다. 핵심 트렌드를 짚어내고 정보를 종합하는 데 능숙합니다. "
    "요청받은 주제에 대해 가장 중요한 트렌드 3가지를 번호 목록으로 정리하세요. "
    "각 항목은 실용적인 활용과 잠재적 영향에 초점을 맞추세요."
)

WRITER_ROLE = (
    "당신은 기술 콘텐츠 작가입니다. 복잡한 기술 주제를 일반 독자가 이해하기 쉬운 글로 풀어내는 데 능숙합니다. "
    "리서치 결과를 바탕으로 읽기 쉽고 흥미로운 블로그 글을 작성하세요. "
    "편집자 피드백이 주어지면 반드시 반영하세요."
)

EDITOR_ROLE = (
    "당신은 블로그 편집장입니다. 초안이 일반 독자에게 명확하고 흥미로운지 검토하세요. "
    "출판 가능하면 'APPROVED'만 출력하세요. "
    "수정이 필요하면 첫 줄에 'REVISE'를 쓰고, 그 아래에 구체적인 수정 요청을 글머리 기호로 나열하세요."
)


# ---------------------------------------------------------------------------
# 2. 상태(State) 정의
#    에이전트 간에 넘겨주는 산출물을 전부 상태의 키로 둔다. (= 핸드오프 채널)
# ---------------------------------------------------------------------------
class BlogCrewState(TypedDict):
    topic: str               # 블로그 주제
    research_findings: str   # Researcher → Writer 전달물
    draft: str               # Writer → Editor 전달물
    feedback: str            # Editor → Writer 전달물 (수정 요청 시)
    revision_count: int      # 수정 요청 횟수
    final_post: str          # 승인된 최종 블로그


# ---------------------------------------------------------------------------
# 3. Fake 모델 정의
#    세 에이전트가 같은 모델을 공유하므로, 그래프의 호출 순서대로 응답을 배치한다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # [Researcher] 트렌드 조사 결과
    """1. 에이전틱 AI: LLM이 도구를 호출하고 여러 단계를 스스로 계획·실행하는 자율 에이전트가 확산 중.
2. 멀티모달 모델: 텍스트·이미지·음성·영상을 하나의 모델이 함께 이해·생성하는 방향으로 발전.
3. 소형 특화 모델(SLM): 비용과 지연 시간을 줄이기 위해 특정 작업에 최적화된 경량 모델 수요 증가.""",

    # [Writer] 초안 v1 (의도적으로 다소 딱딱하게 작성 → 편집자가 수정 요청)
    """# 2025년 주목해야 할 AI 트렌드 3가지

첫째, 에이전틱 AI이다. LLM이 도구를 호출하고 작업을 계획·실행한다.
둘째, 멀티모달 모델이다. 텍스트, 이미지, 음성, 영상을 함께 처리한다.
셋째, 소형 특화 모델이다. 비용과 지연 시간을 줄인다.""",

    # [Editor] 1차 검토: 수정 요청
    """REVISE
- 일반 독자가 읽기에는 문장이 너무 짧고 건조합니다. 각 트렌드에 실생활 예시를 하나씩 넣어주세요.
- 도입부와 마무리 문단이 없습니다. 왜 이 트렌드가 중요한지 한 문단으로 열고 닫아주세요.""",

    # [Writer] 초안 v2 (피드백 반영)
    "# 2025년 주목해야 할 AI 트렌드 3가지\n\n"
    "AI는 이제 \"질문에 답하는 도구\"를 넘어 \"일을 대신 해주는 동료\"로 진화하고 있습니다. "
    "올해 특히 주목할 세 가지 흐름을 소개합니다.\n\n"
    "**1. 에이전틱 AI** — 단순히 답변을 생성하는 데서 그치지 않고, "
    "스스로 계획을 세우고 검색·계산 같은 도구를 호출하며 여러 단계를 처리합니다. "
    "예를 들어 \"다음 주 출장 일정 짜줘\"라고 하면 항공편 검색부터 캘린더 등록까지 알아서 진행하는 식입니다.\n\n"
    "**2. 멀티모달 모델** — 텍스트뿐 아니라 이미지·음성·영상을 한 모델이 함께 이해하고 생성합니다. "
    "냉장고 사진을 찍어 보내면 남은 재료로 만들 수 있는 요리를 추천받는 경험이 가능해집니다.\n\n"
    "**3. 소형 특화 모델(SLM)** — 거대한 범용 모델 대신, 특정 업무에 맞춰 가볍게 만든 모델이 각광받고 있습니다. "
    "스마트폰 안에서 인터넷 없이 실시간 번역이 돌아가는 것이 대표적인 사례입니다.\n\n"
    "이 세 흐름은 모두 \"AI를 더 실용적이고, 더 가깝게\" 만드는 방향을 가리킵니다. "
    "내년에는 이 기술들이 우리 일상 속에 얼마나 자연스럽게 스며들지 지켜볼 만합니다.",

    # [Editor] 2차 검토: 승인
    "APPROVED",
])


# ---------------------------------------------------------------------------
# 4. 헬퍼 함수
# ---------------------------------------------------------------------------
def call_agent(role_prompt: str, content: str) -> str:
    """역할(시스템 프롬프트)을 부여해 LLM을 호출하는 공통 헬퍼"""
    response = llm.invoke([
        SystemMessage(content=role_prompt),
        HumanMessage(content=content),
    ])
    return response.content


# ---------------------------------------------------------------------------
# 5. 에이전트 노드 정의
# ---------------------------------------------------------------------------
def researcher_node(state: BlogCrewState) -> dict:
    """[Researcher] 주제에 대한 최신 트렌드를 조사·요약한다."""
    print("\n>>> [Researcher] AI 트렌드 조사 중...")

    findings = call_agent(
        RESEARCHER_ROLE,
        f"'{state['topic']}'에 대해 2024-2025년의 주요 트렌드 3가지를 조사해 요약하세요.",
    )

    print(f"--- 리서치 결과 ---\n{findings}")
    return {"research_findings": findings, "revision_count": 0, "feedback": ""}


def writer_node(state: BlogCrewState) -> dict:
    """[Writer] 리서치 결과(와 편집자 피드백)를 바탕으로 블로그 초안을 작성한다."""
    is_revision = bool(state.get("feedback"))
    draft_label = "수정본" if is_revision else "초안"
    print(f"\n>>> [Writer] 블로그 {draft_label} 작성 중...")

    request = (
        f"주제: {state['topic']}\n\n"
        f"[리서치 결과]\n{state['research_findings']}\n\n"
        f"위 내용을 바탕으로 일반 독자를 위한 블로그 글을 작성하세요."
    )
    # 재작성인 경우 편집자 피드백을 요청에 포함시킨다. (Editor → Writer 핸드오프)
    if is_revision:
        request += f"\n\n[편집자 피드백 — 반드시 반영]\n{state['feedback']}"

    draft = call_agent(WRITER_ROLE, request)

    print(f"--- 작성된 글 ---\n{draft}")
    return {"draft": draft}


def editor_node(state: BlogCrewState) -> dict:
    """[Editor] 초안을 검토해 승인하거나 수정 요청을 남긴다."""
    print("\n>>> [Editor] 초안 검토 중...")

    verdict = call_agent(
        EDITOR_ROLE,
        f"주제: {state['topic']}\n\n[검토할 초안]\n{state['draft']}",
    )

    if verdict.strip().startswith("APPROVED"):
        print("--- 편집자 판정: 승인 ---")
        return {"final_post": state["draft"], "feedback": ""}

    print(f"--- 편집자 판정: 수정 요청 ---\n{verdict}")
    return {"feedback": verdict, "revision_count": state["revision_count"] + 1}


def route_after_editor(state: BlogCrewState) -> str:
    """[조건부 분기] 편집자 노드 실행 후: 승인됐거나 수정 한도를 넘었으면 종료, 아니면 작가에게 되돌린다."""
    if state.get("final_post"):
        return "approve"
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"--- 수정 한도({MAX_REVISIONS}회) 도달: 현재 초안을 최종본으로 확정 ---")
        return "approve"
    return "revise"


def finalize_node(state: BlogCrewState) -> dict:
    """수정 한도 도달 시 현재 초안을 최종본으로 채워 넣는다. (승인된 경우는 이미 채워져 있음)"""
    return {"final_post": state.get("final_post") or state["draft"]}


# ---------------------------------------------------------------------------
# 6. 그래프 구성
#    START → researcher → writer → editor ─(approve)→ finalize → END
#                              ↑            └(revise)──┘
# ---------------------------------------------------------------------------
graph_builder = StateGraph(BlogCrewState)

graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("writer", writer_node)
graph_builder.add_node("editor", editor_node)
graph_builder.add_node("finalize", finalize_node)

graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "writer")   # Researcher → Writer 핸드오프
graph_builder.add_edge("writer", "editor")       # Writer → Editor 핸드오프
graph_builder.add_conditional_edges(
    "editor",
    route_after_editor,
    {
        "revise": "writer",      # Editor → Writer 핸드오프 (피드백과 함께 루프백)
        "approve": "finalize",
    },
)
graph_builder.add_edge("finalize", END)

blog_crew = graph_builder.compile()


# ---------------------------------------------------------------------------
# 7. 실행
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"## 블로그 제작 팀 실행: '{TOPIC}' ##")

    result = blog_crew.invoke({"topic": TOPIC})

    print("\n\n" + "=" * 30 + " 최종 결과 " + "=" * 30)
    print(f"(편집자 수정 요청 {result['revision_count']}회 반영)\n")
    print(result["final_post"])


if __name__ == "__main__":
    main()
