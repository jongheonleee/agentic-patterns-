"""
13. Human-in-the-Loop 패턴

에이전트가 모든 것을 혼자 결정하게 두지 않고, 중요한 지점에서 사람에게 실행을 멈추고 물어본 뒤
그 판단(승인 · 반려 · 피드백)을 받아 이어서 진행하는 패턴.
에이전트의 속도·규모와 사람의 판단력·책임을 결합한다.

    개입 지점 (Checkpoint)   : 위험하거나 확신이 낮은 결정, 되돌리기 어려운 행동 앞에 "사람 확인" 단계를 둔다.
    일시 중지 (Interrupt)    : 그 지점에서 그래프 실행을 멈추고, 사람이 판단할 정보(요약 · 제안)를 내보낸다.
    재개 (Resume)            : 사람의 응답을 받아 멈춘 자리에서 그대로 이어서 실행한다. 사람의 판단이 이후 흐름을 바꾼다.
    개인화 (Personalization) : 고객 정보(이름 · 등급 · 구매 이력 · 상담 이력)를 프롬프트에 주입해 응대를 맞춤화한다.

이 예제는 "전자제품 회사의 기술 지원 에이전트"다. 이 예제에는 두 종류의 사람이 개입한다.
    - 상담원(전문가) : 에이전트가 "복잡한 문제라 전문가 이관이 필요하다"고 판단하면, 이관 전에 승인/반려한다.
    - 고객           : 에이전트가 안내한 해결 단계로 문제가 해결되었는지 답한다. 해결되지 않았으면 티켓을 만든다.

흐름:
    M1. 고객 문의 입력 + 고객 정보(customer_info) 로드
      → H1. personalize   : 고객 정보로 개인화 지침 생성 (ADK 의 before_model_callback 역할)
      → H2. triage        : LLM 이 문의를 분류 → 기본 문제 해결(troubleshoot) / 전문가 이관(escalate) 제안
            ├─ escalate    → [사람 개입 ①] H3. human_review : 상담원이 이관 제안을 승인/반려 (interrupt)
            │                    ├─ 승인 → H4. escalate_to_human 도구 → M2
            │                    └─ 반려 → H5로 (상담원 메모와 함께 기본 해결 절차 진행)
            └─ troubleshoot → H5. troubleshoot : troubleshoot_issue 도구로 해결 단계 안내
                             → [사람 개입 ②] H6. customer_check : 고객이 해결 여부 응답 (interrupt)
                                  ├─ 해결   → M2
                                  └─ 미해결 → H7. create_ticket 도구 → M2
      → M2. respond : 고객 정보 + 처리 결과로 개인화된 최종 답변 생성

왜 쓰는가:
    - LLM 의 판단은 틀릴 수 있다. 화재 위험·고비용 수리처럼 실수의 대가가 큰 결정은 사람이 최종 확인해야 한다.
    - 사람의 판단이 기록되고 흐름에 반영되므로 책임 소재가 분명하고, 에이전트의 오판도 바로잡을 수 있다.
      (예제 4번: 에이전트는 "메인보드 고장"이라는 말에 이관을 제안하지만, 상담원이 반려하고 기본 절차로 돌린다.)
    - "해결되었나요?"처럼 사람만 알 수 있는 정보는 추측하지 말고 물어보는 것이 정확하다.

LangGraph 사용 포인트:
    - interrupt(payload) : 노드 안에서 호출하면 그래프가 멈추고 payload 가 invoke 결과의 "__interrupt__" 로 나온다.
    - Command(resume=값) : 같은 thread_id 로 invoke 하면 멈춘 interrupt() 가 그 값을 반환하며 실행이 이어진다.
    - checkpointer(InMemorySaver) : 멈춘 시점의 상태를 thread_id 로 저장해 두어야 재개할 수 있다. (HITL 의 필수 조건)
    - 사람의 응답은 상태(review_decision / resolved)에 저장되고, 조건부 엣지가 그 값을 보고 다음 노드를 고른다.

원래 Google ADK 스케치와의 대응:
    - Agent(instruction + tools) 하나에 맡기던 판단을 triage / troubleshoot / escalate / create_ticket 노드로 나눠
      흐름을 그래프로 명시했다. 세 도구(troubleshoot_issue, create_ticket, escalate_to_human)는 그대로 쓴다.
    - personalization_callback(LLM 요청 직전에 고객 정보를 시스템 메시지로 끼워 넣는 콜백)은 H1 personalize 노드가
      만든 개인화 지침을 각 LLM 호출의 SystemMessage 에 붙이는 방식으로 옮겼다.
    - ADK 에서는 escalate_to_human 이 "사람에게 넘겼다"는 메시지만 돌려주지만,
      여기서는 interrupt 로 실제로 실행을 멈추고 사람의 응답을 받아 이어간다.

학습용이므로 실제 API 대신 규칙 기반 Fake 모델을 사용한다. (문의 내용의 키워드로 분류·답변)
아래 예시 문의를 입력하면 각 경로를 직접 확인할 수 있다.
    1) 와이파이가 자꾸 끊겨요                       → 기본 해결 안내 → 고객: 해결됨
    2) 노트북 배터리가 너무 빨리 닳아요             → 기본 해결 안내 → 고객: 미해결 → 티켓 생성
    3) 노트북에서 타는 냄새가 나고 연기가 나요       → 이관 제안 → 상담원: 승인 → 전문가 이관
    4) 화면이 깜빡이는데 메인보드 고장인가요?       → 이관 제안 → 상담원: 반려 → 기본 해결 안내 → 고객: 해결됨
`python 13-human-in-the-loop.py --demo` 로 실행하면 위 예시를 정해진 사람 응답으로 자동 실행한다.
"""
import json
import re
import sys
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 고객 정보: 실제로는 CRM 에서 조회한다. ADK 스케치의 state["customer_info"] 에 해당한다.
CUSTOMER_INFO = {
    "name": "김지수",
    "tier": "골드",
    "recent_purchases": ["울트라북 14", "무선 이어폰 프로"],
    "support_history": ["2026-10-02 울트라북 14 충전기 불량 → 교환 완료"],
}

# 도구(troubleshoot_issue)가 돌려줄 문제 유형별 기본 해결 단계
TROUBLESHOOT_GUIDES = {
    "네트워크": [
        "공유기 전원을 30초간 껐다 켜기",
        "기기의 Wi-Fi 를 끄고 다시 연결하기",
        "네트워크 드라이버 업데이트하기",
    ],
    "전원/배터리": [
        "화면 밝기를 낮추고 백그라운드 앱 종료하기",
        "전원 관리 모드를 '균형'으로 바꾸기",
        "배터리 상태 진단 실행하기",
    ],
    "디스플레이": [
        "그래픽 드라이버 업데이트하기",
        "화면 주사율을 기본값(60Hz)으로 되돌리기",
        "외부 모니터에서도 같은 증상인지 확인하기",
    ],
    "일반": [
        "기기를 완전히 종료한 뒤 다시 켜기",
        "최신 시스템 업데이트 설치하기",
    ],
}

# Fake 모델이 문의를 분류할 때 쓰는 키워드
ISSUE_TYPE_KEYWORDS = {
    "네트워크": r"와이파이|wi-?fi|인터넷|네트워크|공유기",
    "전원/배터리": r"배터리|충전|전원",
    "디스플레이": r"화면|디스플레이|모니터|깜빡",
}
# 위험하거나 기본 절차로 해결할 수 없는 신호
ESCALATION_KEYWORDS = r"타는 냄새|연기|불꽃|화재|부풀|침수|메인보드|감전"

# 역할 프롬프트
TRIAGE_ROLE = (
    "당신은 전자제품 회사의 기술 지원 전문가입니다. 고객 문의를 분류하세요. "
    "기본 문제 해결로 풀 수 있으면 troubleshoot, 위험하거나 기본 절차를 넘어서는 문제면 escalate 를 고르세요. "
    '반드시 {"action": ..., "issue_type": ..., "reason": ...} JSON 으로만 답하세요.'
)

RESPOND_ROLE = (
    "당신은 전자제품 회사의 기술 지원 전문가입니다. 처리 결과를 고객에게 전달하세요. "
    "기술 문제가 주는 불편에 공감하되, 전문적이고 명확하게 다음 단계를 안내하세요."
)

# 사람 응답 해석: 이 중 하나를 입력하면 "예"로 본다.
APPROVE_ANSWERS = {"y", "yes", "예", "승인"}    # 상담원: 이관 승인
RESOLVED_ANSWERS = {"y", "yes", "예", "네"}     # 고객: 문제 해결됨

# --demo 모드: (고객 문의, 사람 응답 목록). 응답은 interrupt 가 발생하는 순서대로 쓰인다.
DEMO_CASES = [
    ("와이파이가 자꾸 끊겨요", [True]),
    ("노트북 배터리가 너무 빨리 닳아요", [False]),
    (
        "노트북에서 타는 냄새가 나고 연기가 나요",
        [
            {
                "approve": True,
                "note": "화재 위험이 있으니 즉시 전원을 끄고 사용을 중지해 주세요. 방문 수리를 예약해 드리겠습니다",
            },
        ],
    ),
    (
        "화면이 깜빡이는데 메인보드 고장인가요?",
        [
            {
                "approve": False,
                "note": "메인보드보다는 드라이버·주사율 문제일 가능성이 높음",
            },
            True,
        ],
    ),
]


# ---------------------------------------------------------------------------
# 2. 상태(State) 정의
#    사람의 응답(review_decision, resolved)도 상태에 저장되어 이후 흐름을 결정한다.
# ---------------------------------------------------------------------------
class SupportState(TypedDict, total=False):
    issue: str                    # 고객 문의
    customer_info: dict           # 고객 정보
    personalization: str          # H1 에서 만든 개인화 지침 (모든 LLM 호출의 SystemMessage 에 붙는다)
    triage: dict                  # H2 분류 결과 {"action", "issue_type", "reason"}
    review_decision: str          # H3 상담원 판단: approve / reject
    reviewer_note: str            # H3 상담원 메모
    steps: list[str]              # H5 안내한 해결 단계
    resolved: bool                # H6 고객 응답: 해결 여부
    ticket_id: str                # H7 생성된 티켓 번호
    outcome: str                  # 최종 처리 결과: resolved / ticket / escalated
    response: str                 # 고객에게 보낼 최종 답변


# ---------------------------------------------------------------------------
# 3. 도구 정의
#    ADK 스케치의 세 도구를 그대로 옮기고, 학습용 데이터만 채웠다.
# ---------------------------------------------------------------------------
_ticket_counter = 0


def troubleshoot_issue(issue_type: str) -> dict:
    """[도구] 문제 유형별 기본 해결 단계를 돌려준다."""
    steps = TROUBLESHOOT_GUIDES.get(issue_type, TROUBLESHOOT_GUIDES["일반"])
    return {"status": "success", "steps": steps}


def create_ticket(issue_type: str, details: str) -> dict:
    """[도구] 기본 해결로 풀리지 않은 문제를 지원 티켓으로 등록한다."""
    global _ticket_counter
    _ticket_counter += 1

    return {
        "status": "success",
        "ticket_id": f"TICKET-{_ticket_counter:04d}",
        "issue_type": issue_type,
        "details": details,
    }


def escalate_to_human(issue_type: str, note: str) -> dict:
    """[도구] 문제를 사람 전문가에게 이관한다. 상담원 메모를 함께 넘긴다."""
    return {
        "status": "success",
        "message": f"'{issue_type}' 문제를 전문가에게 이관했습니다.",
        "note": note,
    }


# ---------------------------------------------------------------------------
# 4. Fake 모델 정의
#    사용자가 무엇을 입력할지 모르므로, 요청에 담긴 JSON 을 읽어 규칙대로 답하는 Fake 모델을 쓴다.
#    task 가 "triage" 면 문의를 분류하고, "respond" 면 처리 결과로 고객 답변을 쓴다.
#    실제 서비스에서는 이 자리에 ChatOpenAI / ChatAnthropic 등을 넣으면 된다.
# ---------------------------------------------------------------------------
class FakeSupportLLM(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-support-llm"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        payload = json.loads(messages[-1].content)

        if payload["task"] == "triage":
            return self._triage(payload["issue"])
        return self._respond(payload, messages[0].content)

    @staticmethod
    def _triage(issue: str) -> str:
        issue_type = "일반"
        for candidate, pattern in ISSUE_TYPE_KEYWORDS.items():
            if re.search(pattern, issue, re.I):
                issue_type = candidate
                break

        danger = re.search(ESCALATION_KEYWORDS, issue)
        if danger:
            result = {
                "action": "escalate",
                "issue_type": issue_type,
                "reason": f"'{danger.group()}' 언급 — 기본 절차로 해결하기 어렵거나 위험할 수 있음",
            }
            return json.dumps(result, ensure_ascii=False)

        result = {
            "action": "troubleshoot",
            "issue_type": issue_type,
            "reason": "기본 해결 절차로 해결 가능한 문제로 보임",
        }
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _respond(payload: dict, system_prompt: str) -> str:
        # 시스템 프롬프트에 주입된 개인화 지침에서 고객 이름·등급·이력을 읽는다.
        name = re.search(r"고객 이름: (.+)", system_prompt)
        tier = re.search(r"고객 등급: (.+)", system_prompt)
        history = re.search(r"상담 이력: (.+)", system_prompt)

        if name:
            greeting = f"{name.group(1)}님({tier.group(1)} 등급), 불편을 드려 죄송합니다."
        else:
            greeting = "불편을 드려 죄송합니다."
        if history:
            greeting += f" 지난 상담 이력({history.group(1)})도 함께 확인했습니다."

        outcome = payload["outcome"]
        if outcome == "escalated":
            body = "이 문제는 전문가가 직접 확인해야 하는 사안이라 담당 전문가에게 연결해 드렸습니다."
            if payload.get("note"):
                body += f" 상담원 안내: {payload['note']}."
        elif outcome == "ticket":
            body = (
                f"안내드린 단계로 해결되지 않아 지원 티켓 {payload['ticket_id']} 을(를) 등록했습니다. "
                "담당자가 1영업일 안에 연락드리겠습니다."
            )
        else:
            body = "안내드린 단계로 문제가 해결되어 다행입니다. 같은 증상이 다시 나타나면 언제든 문의해 주세요."

        return f"{greeting} {body}"


llm = FakeSupportLLM()


# ---------------------------------------------------------------------------
# 5. 헬퍼 함수
# ---------------------------------------------------------------------------
def system_prompt(role: str, state: SupportState) -> SystemMessage:
    """역할 프롬프트 뒤에 개인화 지침을 붙인다. (ADK personalization_callback 이 하던 일)"""
    return SystemMessage(content=role + "\n" + state.get("personalization", ""))


# ---------------------------------------------------------------------------
# 6. 노드 정의
# ---------------------------------------------------------------------------
def personalize(state: SupportState) -> dict:
    """[H1] 고객 정보를 개인화 지침 문자열로 만든다."""
    info = state.get("customer_info") or {}
    if not info:
        return {"personalization": ""}

    lines = [
        "[개인화 지침]",
        f"고객 이름: {info.get('name', '고객')}",
        f"고객 등급: {info.get('tier', '일반')}",
    ]
    if info.get("recent_purchases"):
        lines.append(f"최근 구매: {', '.join(info['recent_purchases'])}")
    if info.get("support_history"):
        lines.append(f"상담 이력: {'; '.join(info['support_history'])}")
    note = "\n".join(lines)

    print("\n[H1] personalize: 고객 정보 주입")
    print("    " + note.replace("\n", "\n    "))
    return {"personalization": note}


def triage(state: SupportState) -> dict:
    """[H2] LLM 이 문의를 분류하고, 기본 해결 / 전문가 이관 중 하나를 제안한다."""
    request = json.dumps(
        {"task": "triage", "issue": state["issue"]},
        ensure_ascii=False,
    )

    response = llm.invoke([
        system_prompt(TRIAGE_ROLE, state),
        HumanMessage(content=request),
    ])
    result = json.loads(response.content)

    print(f"\n[H2] triage: 유형={result['issue_type']} / 제안={result['action']} ({result['reason']})")
    return {"triage": result}


def human_review(state: SupportState) -> dict:
    """[H3] 사람 개입 ① — 전문가 이관 전에 상담원의 승인을 받는다. interrupt() 에서 실행이 멈춘다.

    주의: 재개되면 이 노드는 interrupt() 지점이 아니라 "노드의 처음부터" 다시 실행된다.
    그래서 interrupt() 앞에는 두 번 실행되어도 괜찮은 코드만 둔다. (출력·도구 호출·DB 쓰기는 interrupt() 뒤에)
    """
    answer = interrupt({
        "type": "human_review",
        "question": "에이전트가 전문가 이관을 제안했습니다. 승인하시겠습니까?",
        "issue": state["issue"],
        "triage": state["triage"],
    })
    # 재개되면 interrupt() 가 Command(resume=...) 로 넘긴 값을 돌려준다.

    if answer["approve"]:
        decision = "approve"
        decision_label = "승인"
    else:
        decision = "reject"
        decision_label = "반려"

    note_text = ""
    if answer["note"]:
        note_text = f" / 메모: {answer['note']}"

    print(f"\n[H3] human_review: 상담원 판단 = {decision_label}{note_text}")
    return {"review_decision": decision, "reviewer_note": answer["note"]}


def escalate(state: SupportState) -> dict:
    """[H4] 승인된 이관을 실행한다."""
    result = escalate_to_human(
        state["triage"]["issue_type"],
        state.get("reviewer_note", ""),
    )
    print(f"\n[H4] escalate_to_human: {result['message']}")
    return {"outcome": "escalated"}


def troubleshoot(state: SupportState) -> dict:
    """[H5] 기본 해결 단계를 조회해 고객에게 안내한다."""
    issue_type = state["triage"]["issue_type"]
    result = troubleshoot_issue(issue_type)

    print(f"\n[H5] troubleshoot_issue: '{issue_type}' 해결 단계 안내")
    if state.get("review_decision") == "reject" and state.get("reviewer_note"):
        print(f"    (상담원 메모 반영: {state['reviewer_note']})")
    for i, step in enumerate(result["steps"], start=1):
        print(f"    {i}. {step}")

    return {"steps": result["steps"]}


def customer_check(state: SupportState) -> dict:
    """[H6] 사람 개입 ② — 안내한 단계로 해결되었는지 고객에게 묻는다. 사람만 아는 정보는 추측하지 않는다."""
    # 재개 시 노드가 처음부터 다시 실행되므로 앞에 부수 효과를 두지 않는다
    resolved = interrupt({
        "type": "customer_check",
        "question": "안내해 드린 단계로 문제가 해결되었나요?",
        "steps": state["steps"],
    })

    if resolved:
        resolved_label = "해결됨"
        outcome = "resolved"
    else:
        resolved_label = "해결 안 됨"
        outcome = "ticket"

    print(f"\n[H6] customer_check: 고객 응답 = {resolved_label}")
    return {"resolved": resolved, "outcome": outcome}


def open_ticket(state: SupportState) -> dict:
    """[H7] 해결되지 않은 문제를 티켓으로 등록한다."""
    result = create_ticket(state["triage"]["issue_type"], state["issue"])
    print(f"\n[H7] create_ticket: {result['ticket_id']} 등록")
    return {"ticket_id": result["ticket_id"]}


def respond(state: SupportState) -> dict:
    """[M2] 처리 결과와 개인화 지침으로 최종 답변을 만든다."""
    print("\n[M2] respond: 최종 답변 생성")

    payload = {
        "task": "respond",
        "issue": state["issue"],
        "outcome": state["outcome"],
        "ticket_id": state.get("ticket_id", ""),
        "note": state.get("reviewer_note", ""),
    }
    request = json.dumps(payload, ensure_ascii=False)

    response = llm.invoke([
        system_prompt(RESPOND_ROLE, state),
        HumanMessage(content=request),
    ])
    return {"response": response.content}


# 라우팅 함수
#    LLM 의 제안(triage)과 사람의 판단(review_decision, resolved)이 각각 분기를 결정한다.
def route_after_triage(state: SupportState) -> str:
    if state["triage"]["action"] == "escalate":
        return "review"
    return "troubleshoot"


def route_after_review(state: SupportState) -> str:
    if state["review_decision"] == "approve":
        return "escalate"
    return "troubleshoot"


def route_after_check(state: SupportState) -> str:
    if state["resolved"]:
        return "respond"
    return "ticket"


# ---------------------------------------------------------------------------
# 7. 그래프 구성
#    interrupt 로 멈췄다가 재개하려면 checkpointer 가 반드시 필요하다.
# ---------------------------------------------------------------------------
graph_builder = StateGraph(SupportState)

graph_builder.add_node("personalize", personalize)
graph_builder.add_node("triage", triage)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("escalate", escalate)
graph_builder.add_node("troubleshoot", troubleshoot)
graph_builder.add_node("customer_check", customer_check)
graph_builder.add_node("create_ticket", open_ticket)
graph_builder.add_node("respond", respond)

graph_builder.add_edge(START, "personalize")
graph_builder.add_edge("personalize", "triage")
graph_builder.add_conditional_edges(
    "triage",
    route_after_triage,
    {
        "review": "human_review",
        "troubleshoot": "troubleshoot",
    },
)
graph_builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "escalate": "escalate",
        "troubleshoot": "troubleshoot",   # 사람이 반려하면 기본 절차로
    },
)
graph_builder.add_edge("escalate", "respond")
graph_builder.add_edge("troubleshoot", "customer_check")
graph_builder.add_conditional_edges(
    "customer_check",
    route_after_check,
    {
        "respond": "respond",
        "ticket": "create_ticket",
    },
)
graph_builder.add_edge("create_ticket", "respond")
graph_builder.add_edge("respond", END)

support_agent = graph_builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 8. 사람 응답 받기
#    대화형 모드에서는 터미널에서 입력받고, --demo 모드에서는 미리 정한 응답을 쓴다.
# ---------------------------------------------------------------------------
def ask_human(request: dict) -> Any:
    """interrupt payload 를 보여주고 사람의 응답을 받는다. 반환값이 Command(resume=...) 로 그래프에 전달된다."""
    print(f"    ┌─ 👤 사람 확인 필요: {request['question']}")

    if request["type"] == "human_review":
        print(f"    │  문의: {request['issue']}")
        print(f"    │  에이전트 판단 근거: {request['triage']['reason']}")
        reply = input("    └─ [상담원] 승인(y) / 반려(n)> ").strip().lower()
        approve = reply in APPROVE_ANSWERS
        note = input("       [상담원] 메모(선택, Enter 생략)> ").strip()
        return {"approve": approve, "note": note}

    reply = input("    └─ [고객] 해결됨(y) / 해결 안 됨(n)> ").strip().lower()
    return reply in RESOLVED_ANSWERS


def scripted_human(answers: list) -> Any:
    """--demo 용: 정해진 응답을 차례로 돌려준다."""
    def respond_with(request: dict) -> Any:
        answer = answers.pop(0)
        print(f"    ┌─ 👤 사람 확인 필요: {request['question']}")

        if isinstance(answer, dict):
            if answer["approve"]:
                shown = "승인"
            else:
                shown = "반려"
            if answer["note"]:
                shown += f" / 메모: {answer['note']}"
        elif answer:
            shown = "해결됨"
        else:
            shown = "해결 안 됨"

        print(f"    └─ (데모 응답) {shown}")
        return answer

    return respond_with


# ---------------------------------------------------------------------------
# 9. 실행
#    invoke → "__interrupt__" 가 있으면 사람에게 묻고 Command(resume=...) 로 재개 → 끝날 때까지 반복
# ---------------------------------------------------------------------------
_thread_seq = 0


def handle_issue(issue: str, human=ask_human) -> None:
    global _thread_seq
    _thread_seq += 1
    thread_id = f"support-{_thread_seq}"
    config = {"configurable": {"thread_id": thread_id}}   # 멈춘 상태를 찾아 재개하는 키

    print(f"\n[M1] 고객 문의: {issue}")
    initial_state = {"issue": issue, "customer_info": CUSTOMER_INFO}
    result = support_agent.invoke(initial_state, config)

    while "__interrupt__" in result:                       # 사람 확인 지점에서 멈춤
        request = result["__interrupt__"][0].value
        if request["type"] == "human_review":
            stage = "H3 human_review"
        else:
            stage = "H6 customer_check"
        print(f"\n⏸  [{stage}] interrupt — 실행 일시 중지, 상태는 checkpointer 에 저장됨")

        answer = human(request)

        print(f"▶  Command(resume=...) — thread_id '{thread_id}' 로 재개")
        result = support_agent.invoke(Command(resume=answer), config)   # 같은 thread_id 로 재개

    print(f"\n[답변] {result['response']}")


def main() -> None:
    print("## Human-in-the-Loop 기술 지원 에이전트 실행 ##")

    if "--demo" in sys.argv:
        for issue, answers in DEMO_CASES:
            print("\n" + "=" * 80)
            handle_issue(issue, human=scripted_human(list(answers)))
        return

    customer = f"{CUSTOMER_INFO['name']} ({CUSTOMER_INFO['tier']} 등급)"
    print(f"고객: {customer}. 문의 내용을 입력하세요. (종료: q)")
    examples = [issue for issue, _ in DEMO_CASES]
    print("예시:\n    " + "\n    ".join(examples))

    while True:
        try:
            issue = input("\n문의> ").strip()
            if issue.lower() in {"q", "quit", "exit"}:
                break
            if issue:
                handle_issue(issue)
        except (EOFError, KeyboardInterrupt):             # Ctrl+D / Ctrl+C 도 오류 없이 종료
            print()
            break

    print("에이전트를 종료합니다.")


if __name__ == "__main__":
    main()
