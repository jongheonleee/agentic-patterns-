"""
12. Exception Handling and Recovery 패턴

에이전트가 도구(외부 API)를 쓰다 보면 타임아웃, 서버 장애, 잘못된 입력, "결과 없음" 같은 실패를 반드시 만난다.
이 패턴은 실패를 "프로그램이 죽는 사건"이 아니라 "에이전트가 다루어야 할 상태"로 보고,
감지 → 처리 → 복구 단계를 거쳐 사용자에게 항상 쓸모 있는 응답을 돌려주도록 설계하는 패턴.

    오류 감지 (Detection) : 도구 호출을 try/except 로 감싸 예외를 잡고, 그 종류를 분류한다.
                            - 일시적 오류 (TimeoutError, ConnectionError) : 다시 시도하면 성공할 수 있다
                            - 영구적 오류 (LookupError, ValueError)       : 몇 번을 다시 해도 결과가 같다
    오류 처리 (Handling)  : 분류에 따라 대응을 고른다.
                            - 재시도 (Retry)   : 일시적 오류만 지수 백오프로 몇 번 더 시도한다
                            - 폴백 (Fallback)  : 영구적 오류이거나 재시도를 다 써도 실패하면 대체 수단으로 넘어간다
                            - 기록 (Logging)   : 어떤 단계에서 무슨 오류가 났고 어떻게 대응했는지 상태에 남긴다
    복구 (Recovery)       : 정밀한 결과 → 대략적인 결과 → 정중한 사과 순으로 "품위 있게 성능을 낮춘다"
                            (Graceful Degradation). 모든 수단이 실패해도 에이전트는 죽지 않고 답한다.

이 예제는 "사용자가 입력한 주소의 위치 정보를 알려주는 에이전트"다.

흐름:
    M1. 사용자 프롬프트 입력
      → [오류 처리 구간] E1. primary_handler : 주소 추출 → 정밀 위치 조회 도구 호출
                       → E2. 오류 감지·분류 (call_with_retry)
                              ├─ 일시적 오류 & 재시도 남음 → 지수 백오프 후 E1 재시도 (최대 MAX_RETRIES 회)
                              ├─ 성공                     → 폴백을 건너뛰고 M2로 이동
                              └─ 영구적 오류 / 재시도 소진 → primary_location_failed = True
                       → E3. fallback_handler : 프롬프트에서 도시 추출 → 지역 개요 조회 도구 호출
                              ├─ 성공 → 대략적인 결과(precision = "general")로 복구
                              └─ 실패 → 결과 없음 (오류만 기록)
      → M2. response_agent : 결과(정밀 / 대략 / 없음)에 맞춰 답변 생성. 결과가 없으면 사과 + 다음 행동 안내
      → M3. 답변과 오류 기록(error_log) 출력

왜 쓰는가:
    - 외부 도구는 언제든 실패한다. 예외를 잡지 않으면 에이전트 전체가 멈추고 사용자는 아무것도 받지 못한다.
    - 오류를 종류별로 나눠야 올바르게 대응할 수 있다. "주소가 없다"는 오류를 재시도하는 것은 시간 낭비이고,
      "잠깐 타임아웃"을 곧바로 포기하는 것은 성공할 기회를 버리는 것이다.
    - 오류 기록(error_log)이 상태에 남아 있으면 응답 에이전트가 "왜 대략적인 결과인지"를 설명할 수 있고,
      개발자는 어떤 실패가 자주 나는지 추적할 수 있다.

LangGraph 사용 포인트:
    - 에이전트 하나 = 노드 하나 (primary_handler / fallback_handler / response_agent).
    - 노드 안에서 예외를 잡아 "실패 플래그 + 오류 기록"을 상태로 돌려준다. 예외가 그래프 밖으로 새지 않는다.
    - primary_location_failed 플래그를 보고 조건부 엣지로 "폴백" 또는 "바로 응답"을 고른다.
    - error_log 는 Annotated[list, operator.add] 로 선언해, 여러 노드가 남긴 기록이 덮어쓰이지 않고 누적된다.

원래 Google ADK 스케치와의 대응:
    ADK 의 SequentialAgent(primary_handler → fallback_handler → response_agent) 구조를 그대로 옮겼다.
    ADK 에서는 fallback_handler 가 항상 실행되고 지시문 안에서 state["primary_location_failed"] 를 확인하지만,
    여기서는 같은 판단을 조건부 엣지로 코드에 명시해, 성공하면 폴백 노드를 아예 건너뛴다.

학습용이므로 실제 API 대신 Fake 도구와 Fake 모델을 사용한다.
Fake 도구는 주소마다 정해진 장애 시나리오(FAULT_SCENARIOS)에 따라 예외를 던지므로,
아래 예시 프롬프트를 입력하면 각 복구 경로를 직접 확인할 수 있다.
    1) 서울특별시 중구 세종대로 110 위치 알려줘       → 한 번에 성공
    2) 서울특별시 종로구 사직로 161 위치 알려줘       → 타임아웃 1회 → 재시도 성공
    3) 부산광역시 해운대구 해운대해변로 999 어디야?   → 주소 없음(영구적) → 폴백으로 부산 개요
    4) 제주특별자치도 제주시 첨단로 242 위치 알려줘  → 서버 장애 지속 → 재시도 소진 → 폴백으로 제주 개요
    5) 화성 올림푸스 산 위치 알려줘                 → 정밀 조회·폴백 모두 실패 → 사과 + 안내
`python 12-exception-handling-and-recovery.py --demo` 로 실행하면 위 예시를 차례로 자동 실행한다.
"""
import json
import operator
import re
import sys
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 오류 분류: 예외를 "다시 시도할 가치가 있는가"로 나눈다. 이 분류가 이후의 모든 대응(재시도 / 폴백)을 결정한다.
TRANSIENT_ERRORS = (TimeoutError, ConnectionError)   # 일시적 오류: 재시도하면 성공할 수 있다
PERMANENT_ERRORS = (LookupError, ValueError)         # 영구적 오류: 재시도해도 결과가 같다 → 즉시 폴백

# 재시도 정책
MAX_RETRIES = 2              # 첫 시도 이후 추가로 재시도할 최대 횟수
BASE_BACKOFF_SECONDS = 0.2   # 지수 백오프의 기본 대기 시간 (0.2s → 0.4s → ...)

# Fake 지도 API 의 데이터
PRECISE_LOCATION_DB = {
    "서울특별시 중구 세종대로 110": {"name": "서울특별시청", "lat": 37.5663, "lng": 126.9779},
    "서울특별시 종로구 사직로 161": {"name": "경복궁", "lat": 37.5796, "lng": 126.9770},
    "제주특별자치도 제주시 첨단로 242": {"name": "제주첨단과학기술단지", "lat": 33.4504, "lng": 126.5704},
}

GENERAL_AREA_DB = {
    "서울": {
        "center": (37.5665, 126.9780),
        "summary": "대한민국의 수도로, 한강을 사이에 두고 25개 자치구로 이루어져 있습니다.",
    },
    "부산": {
        "center": (35.1796, 129.0756),
        "summary": "대한민국 제2의 도시이자 최대 항구 도시로, 해운대·광안리 해변이 유명합니다.",
    },
    "제주": {
        "center": (33.4996, 126.5312),
        "summary": "대한민국 최남단의 화산섬으로, 한라산과 올레길로 유명한 관광지입니다.",
    },
}

# 장애 시나리오: FAULT_SCENARIOS[주소] = n번째 호출에서 던질 예외 목록.
# 목록을 다 쓰면 그다음 호출부터는 정상 동작한다.
FAULT_SCENARIOS: dict[str, list[type[Exception]]] = {
    "서울특별시 종로구 사직로 161": [TimeoutError],                              # 1회만 타임아웃 → 재시도로 복구
    "제주특별자치도 제주시 첨단로 242": [ConnectionError] * (MAX_RETRIES + 1),  # 재시도를 모두 써도 실패
}

# 프롬프트에서 "위치 알려줘", "어디야?" 같은 요청 문구를 떼어 내는 정규식
REQUEST_SUFFIX = re.compile(r"\s*(의)?\s*(위치|좌표|어디)\S*(\s+\S+)?\s*[?？.!]*\s*$")
CITY_SUFFIX = re.compile(r"(특별자치도|특별자치시|특별시|광역시|시|도)$")

RESPONSE_ROLE = (
    "당신은 위치 안내 에이전트입니다. 아래 위치 결과를 사용자에게 명확하고 간결하게 전달하세요. "
    "precision 이 general 이면 정밀한 결과가 아니라는 점을 밝히세요. "
    "결과가 비어 있으면 사과하고, 사용자가 다음에 무엇을 하면 되는지 안내하세요."
)

DEMO_PROMPTS = [
    "서울특별시 중구 세종대로 110 위치 알려줘",
    "서울특별시 종로구 사직로 161 위치 알려줘",
    "부산광역시 해운대구 해운대해변로 999 어디야?",
    "제주특별자치도 제주시 첨단로 242 위치 알려줘",
    "화성 올림푸스 산 위치 알려줘",
]


# ---------------------------------------------------------------------------
# 2. 상태(State) 정의
#    실패를 예외가 아니라 "상태 값"으로 표현하는 것이 이 패턴의 핵심이다.
# ---------------------------------------------------------------------------
class ErrorRecord(TypedDict):
    stage: str        # 오류가 난 단계 (primary_handler / fallback_handler)
    attempt: int      # 몇 번째 시도에서 났는지
    error_type: str   # 예외 클래스 이름
    category: str     # transient / permanent / unknown
    message: str      # 예외 메시지
    action: str       # 그 오류에 대해 에이전트가 취한 대응 (재시도 / 폴백 / 포기)


class LocationAgentState(TypedDict):
    user_prompt: str                                          # 사용자 입력
    location_result: dict                                     # 최종 위치 결과 (비어 있으면 실패)
    primary_location_failed: bool                             # 정밀 조회 실패 여부 → 폴백 분기 조건
    error_log: Annotated[list[ErrorRecord], operator.add]     # 노드들이 남긴 오류 기록 (누적)
    response: str                                             # 사용자에게 보여줄 최종 답변


# ---------------------------------------------------------------------------
# 3. Fake 도구 정의
#    실제 지도 API 대신 작은 데이터베이스(PRECISE_LOCATION_DB)와 장애 시나리오(FAULT_SCENARIOS)를 쓴다.
# ---------------------------------------------------------------------------
_call_counts: dict[str, int] = {}   # 주소별 호출 횟수 (장애 시나리오 재현용)


def reset_fault_simulation() -> None:
    """요청마다 장애 시나리오를 처음부터 다시 재현하도록 호출 횟수를 초기화한다."""
    _call_counts.clear()


def get_precise_location_info(address: str) -> dict:
    """[도구] 주소의 정밀 좌표를 조회한다. 장애 시나리오에 따라 일시적/영구적 예외를 던진다."""
    attempt = _call_counts.get(address, 0)
    _call_counts[address] = attempt + 1

    faults = FAULT_SCENARIOS.get(address, [])
    if attempt < len(faults):
        error_type = faults[attempt]
        raise error_type(f"지도 API {error_type.__name__}: 응답 없음 (address='{address}')")

    if address not in PRECISE_LOCATION_DB:
        raise LookupError(f"등록되지 않은 주소입니다: '{address}'")

    return {"address": address, **PRECISE_LOCATION_DB[address]}


def get_general_area_info(city: str) -> dict:
    """[도구] 도시 단위의 대략적인 정보를 조회한다. (폴백용, 정밀도는 낮지만 안정적)"""
    if city not in GENERAL_AREA_DB:
        raise LookupError(f"지역 정보가 없는 도시입니다: '{city}'")

    area = GENERAL_AREA_DB[city]
    return {
        "city": city,
        "lat": area["center"][0],
        "lng": area["center"][1],
        "summary": area["summary"],
    }


# ---------------------------------------------------------------------------
# 4. Fake 모델 정의
#    다른 예제의 FakeListChatModel 은 정해진 순서로만 답하므로, 사용자가 무엇을 입력할지 모르는 이 예제에는 맞지 않다.
#    그래서 프롬프트에 담긴 위치 결과(JSON)를 읽어 답변을 만드는 "규칙 기반" Fake 모델을 쓴다.
#    실제 서비스에서는 이 자리에 ChatOpenAI / ChatAnthropic 등을 넣으면 된다.
# ---------------------------------------------------------------------------
class FakeResponseLLM(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-response-llm"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        payload = json.loads(messages[-1].content.split("[위치 결과]\n", 1)[1])
        result = payload["location_result"]
        error_count = payload["error_count"]

        # 결과 없음 → 사과 + 다음 행동 안내
        if not result:
            return (
                "죄송합니다. 요청하신 위치 정보를 찾지 못했습니다. "
                "도로명 주소(예: '서울특별시 중구 세종대로 110')나 도시 이름을 포함해 다시 질문해 주세요."
            )

        # 정밀 결과
        if result["precision"] == "precise":
            retry_note = ""
            if error_count:
                retry_note = f" (일시적인 오류 {error_count}건이 있었지만 재시도로 복구했습니다.)"
            return (
                f"'{result['address']}'은(는) {result['name']}의 위치이며, "
                f"좌표는 위도 {result['lat']}, 경도 {result['lng']} 입니다.{retry_note}"
            )

        # 대략적인 결과 (폴백)
        return (
            f"정확한 주소 위치는 확인하지 못해, 대신 '{result['city']}' 지역의 대략적인 정보를 알려드립니다. "
            f"중심 좌표는 위도 {result['lat']}, 경도 {result['lng']} 이며, {result['summary']}"
        )


llm = FakeResponseLLM()


# ---------------------------------------------------------------------------
# 5. 헬퍼 함수
# ---------------------------------------------------------------------------
def classify_error(error: Exception) -> str:
    """[E2 오류 감지] 예외를 transient / permanent / unknown 으로 분류한다."""
    if isinstance(error, TRANSIENT_ERRORS):
        return "transient"
    if isinstance(error, PERMANENT_ERRORS):
        return "permanent"
    return "unknown"         # 예상하지 못한 오류는 재시도하지 않고 폴백으로 넘긴다 (보수적 처리)


def make_error_record(
    stage: str,
    attempt: int,
    error: Exception,
    category: str,
    action: str,
) -> ErrorRecord:
    """오류 하나를 error_log 에 남길 기록으로 만든다."""
    return {
        "stage": stage,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "category": category,
        "message": str(error),
        "action": action,
    }


def call_with_retry(
    tool,
    arg: str,
    stage: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[dict | None, list[ErrorRecord]]:
    """[E2 재시도] 도구 호출을 감싸 예외를 분류하고, 일시적 오류만 지수 백오프로 다시 시도한다.

    결과와 함께 오류 기록을 돌려주므로 호출한 노드가 그대로 상태에 남길 수 있다.
    """
    errors: list[ErrorRecord] = []

    for attempt in range(1, max_retries + 2):
        try:
            return tool(arg), errors
        except Exception as e:                       # 모든 예외를 잡되, 분류해서 다르게 대응한다
            category = classify_error(e)
            can_retry = category == "transient" and attempt <= max_retries

            if can_retry:
                action = "재시도"
            elif category == "transient":
                action = "포기 (재시도 소진)"
            else:
                action = "포기 (재시도 무의미)"

            errors.append(make_error_record(stage, attempt, e, category, action))
            print(f"    ✗ 시도 {attempt}: {type(e).__name__} [{category}] → {action}")

            if not can_retry:
                return None, errors

            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"      [E2] {wait:.1f}초 대기 후 재시도 (지수 백오프)")
            time.sleep(wait)

    return None, errors


def extract_address(prompt: str) -> str:
    """'서울특별시 중구 세종대로 110 위치 알려줘' → '서울특별시 중구 세종대로 110'

    실제로는 LLM 이 프롬프트에서 주소를 뽑아 도구 인자로 넘긴다. 여기서는 규칙으로 대신한다.
    """
    address = REQUEST_SUFFIX.sub("", prompt.strip())
    if not address:
        raise ValueError("프롬프트에서 주소를 찾지 못했습니다.")   # 영구적 오류 → 재시도 없이 폴백
    return address


def extract_city(prompt: str) -> str:
    """프롬프트의 첫 단어에서 도시 이름을 뽑는다. ('부산광역시 ...' → '부산')"""
    words = prompt.split()
    if not words:
        raise ValueError("프롬프트가 비어 있습니다.")
    return CITY_SUFFIX.sub("", words[0])


# ---------------------------------------------------------------------------
# 6. 노드(에이전트) 정의
#    각 노드는 예외를 스스로 처리하고 "결과 / 실패 플래그 / 오류 기록"만 상태로 돌려준다.
# ---------------------------------------------------------------------------
def primary_handler(state: LocationAgentState) -> dict:
    """[E1] 정밀 위치 조회. 일시적 오류는 재시도하고, 끝내 실패하면 실패 플래그를 세운다."""
    print("\n[E1] primary_handler: 정밀 위치 조회")

    try:
        address = extract_address(state["user_prompt"])
    except ValueError as e:
        print(f"    ✗ 입력 해석 실패: {e} → 폴백")
        record = make_error_record(
            stage="primary_handler",
            attempt=1,
            error=e,
            category="permanent",
            action="포기 (재시도 무의미)",
        )
        return {"primary_location_failed": True, "error_log": [record]}

    print(f"    주소 추출: '{address}'")
    result, errors = call_with_retry(get_precise_location_info, address, stage="primary_handler")

    if result is None:
        return {"primary_location_failed": True, "error_log": errors}

    print(f"    ✓ 성공: {result['name']} ({result['lat']}, {result['lng']})")
    return {
        "primary_location_failed": False,
        "location_result": {"precision": "precise", **result},
        "error_log": errors,
    }


def fallback_handler(state: LocationAgentState) -> dict:
    """[E3] 폴백. 정밀도를 낮춰 도시 단위 정보라도 확보한다. 이것마저 실패하면 결과 없이 오류만 남긴다."""
    print("\n[E3] fallback_handler: 정밀 조회 실패 → 도시 단위 정보로 폴백")

    try:
        city = extract_city(state["user_prompt"])
        print(f"    도시 추출: '{city}'")
        result = get_general_area_info(city)
    except Exception as e:
        print(f"    ✗ 폴백 실패: {type(e).__name__}: {e}")
        record = make_error_record(
            stage="fallback_handler",
            attempt=1,
            error=e,
            category=classify_error(e),
            action="포기 (응답 에이전트가 사과)",
        )
        return {"location_result": {}, "error_log": [record]}

    print(f"    ✓ 폴백 성공: {city} 일대 ({result['lat']}, {result['lng']})")
    return {"location_result": {"precision": "general", **result}}


def response_agent(state: LocationAgentState) -> dict:
    """[M2] 결과 수준(정밀 / 대략 / 없음)에 맞춰 답변을 만든다. 실패해도 반드시 답변을 돌려준다."""
    print("\n[M2] response_agent: 답변 생성")

    payload = {
        "location_result": state.get("location_result") or {},
        "error_count": len(state["error_log"]),
    }
    request = (
        f"[사용자 질문]\n{state['user_prompt']}\n\n"
        f"[위치 결과]\n{json.dumps(payload, ensure_ascii=False)}"
    )

    response = llm.invoke([
        SystemMessage(content=RESPONSE_ROLE),
        HumanMessage(content=request),
    ])
    return {"response": response.content}


def route_after_primary(state: LocationAgentState) -> str:
    """primary_location_failed 플래그로 폴백 여부를 결정한다. (ADK 에서는 지시문 안에서 하던 판단)"""
    if state["primary_location_failed"]:
        return "fallback"
    return "respond"


# ---------------------------------------------------------------------------
# 7. 그래프 구성
#    START → primary_handler ─┬─ (실패) → fallback_handler → response_agent → END
#                             └─ (성공) ─────────────────────→ response_agent
# ---------------------------------------------------------------------------
graph_builder = StateGraph(LocationAgentState)

graph_builder.add_node("primary_handler", primary_handler)
graph_builder.add_node("fallback_handler", fallback_handler)
graph_builder.add_node("response_agent", response_agent)

graph_builder.add_edge(START, "primary_handler")
graph_builder.add_conditional_edges(
    "primary_handler",
    route_after_primary,
    {
        "fallback": "fallback_handler",   # 영구적 오류 / 재시도 소진 → 폴백
        "respond": "response_agent",      # 성공 → 폴백 건너뛰고 바로 응답
    },
)
graph_builder.add_edge("fallback_handler", "response_agent")
graph_builder.add_edge("response_agent", END)

robust_location_agent = graph_builder.compile()


# ---------------------------------------------------------------------------
# 8. 실행
#    사용자 입력 → 에이전트 실행 → 답변 + 오류 기록 출력 을 반복한다.
#    그래프 안에서 처리하지 못한 예외가 혹시 새어 나와도 마지막 안전망(try/except)이 잡아 루프가 멈추지 않는다.
# ---------------------------------------------------------------------------
def print_error_log(error_log: list[ErrorRecord]) -> None:
    if not error_log:
        print("[M3] 오류 기록: 없음")
        return

    print(f"[M3] 오류 기록: {len(error_log)}건")
    for r in error_log:
        print(
            f"    - {r['stage']} 시도 {r['attempt']} | {r['error_type']} [{r['category']}] | "
            f"{r['action']} | {r['message']}"
        )


def handle_prompt(prompt: str) -> None:
    print(f"\n[M1] 사용자 프롬프트: {prompt}")
    reset_fault_simulation()

    initial_state = {
        "user_prompt": prompt,
        "location_result": {},
        "primary_location_failed": False,
        "error_log": [],
    }

    try:
        final_state = robust_location_agent.invoke(initial_state)
    except Exception as e:                       # 마지막 안전망: 예상하지 못한 오류도 사용자에게는 정중하게
        print(f"\n[!] 처리 중 예상하지 못한 오류가 발생했습니다: {type(e).__name__}: {e}")
        print("[답변] 죄송합니다. 일시적인 문제로 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        return

    print(f"\n[답변] {final_state['response']}")
    print_error_log(final_state["error_log"])


def main() -> None:
    print("## 예외 처리·복구 위치 안내 에이전트 실행 ##")

    if "--demo" in sys.argv:
        for prompt in DEMO_PROMPTS:
            print("\n" + "=" * 80)
            handle_prompt(prompt)
        return

    print("주소를 포함해 위치를 물어보세요. (종료: q)")
    print("예시:\n    " + "\n    ".join(DEMO_PROMPTS))

    while True:
        try:
            prompt = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):    # Ctrl+D / Ctrl+C 도 오류 없이 종료
            print()
            break

        if prompt.lower() in {"q", "quit", "exit"}:
            break
        if not prompt:
            continue
        handle_prompt(prompt)

    print("에이전트를 종료합니다.")


if __name__ == "__main__":
    main()
