import json 
import operator
import re
import sys
import time

from typing import Annotated, Any, TypedDict
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END 

# 오류 분류 
TRANSIENT_ERRORS = (TimeoutError, ConnectionError)
PERMANENT_ERRORS = (LookupError, ValueError)

MAX_RETRIES = 2
BASE_BACKOFF_SECONDS = 0.2 

def classify_error(error: Exception) -> str:
    
    if isinstance(error, TRANSIENT_ERRORS):
        return "transient"
    
    if isinstance(error, PERMANENT_ERRORS):
        return "permanent"
    
    return "unknown"

# 도구 정의 
PRECISE_LOCATION_DB = {
    "서울특별시 중구 세종대로 110": {"name": "서울특별시청", "lat": 37.5663, "lng": 126.9779},
    "서울특별시 종로구 사직로 161": {"name": "경복궁", "lat": 37.5796, "lng": 126.9770},
    "제주특별자치도 제주시 첨단로 242": {"name": "제주첨단과학기술단지", "lat": 33.4504, "lng": 126.5704},
}

GENERAL_AREA_DB = {
    "서울": {"center": (37.5665, 126.9780), "summary": "대한민국의 수도로, 한강을 사이에 두고 25개 자치구로 이루어져 있습니다."},
    "부산": {"center": (35.1796, 129.0756), "summary": "대한민국 제2의 도시이자 최대 항구 도시로, 해운대·광안리 해변이 유명합니다."},
    "제주": {"center": (33.4996, 126.5312), "summary": "대한민국 최남단의 화산섬으로, 한라산과 올레길로 유명한 관광지입니다."},
}

FAULT_SCENARIOS: dict[str, list[type[Exception]]] = {
    "서울특별시 종로구 사직로 161": [TimeoutError],                            # 1회만 타임아웃 → 재시도로 복구
    "제주특별자치도 제주시 첨단로 242": [ConnectionError] * (MAX_RETRIES + 1),  # 재시도를 모두 써도 실패
}

_call_counts: dict[str, int] = {} # 주소별 호출 횟수 (장애 시나리오 재현용)

def reset_fault_simulation() -> None:
    _call_counts.clear()
    
def get_precise_location_info(address: str) -> dict:
    attempt = _call_counts.get(address, 0)
    _call_counts[address] = attempt + 1
    
    faults = FAULT_SCENARIOS.get(
        address, 
        [],
    )
    
    if attempt < len(faults):
        error_type = faults[attempt]
        raise error_type(f"지도 API {error_type.__name__}: 응답 없음 (address='{address}')")
    
    if address not in PRECISE_LOCATION_DB:
        raise LookupError(f"등록되지 않은 주소입니다: '{address}'")
    
    return {
        "address": address,
        **PRECISE_LOCATION_DB[address]
    }
    
def get_general_area_info(city: str) -> dict:
    """[도구] 도시 단위의 대략적인 정보를 조회한다"""
    
    if city not in GENERAL_AREA_DB:
        raise LookupError(f"지역 정보가 없는 도시입니다: '{city}'")
    
    area = GENERAL_AREA_DB[city]
    
    return {
        "city": city,
        "lat": area["center"][0],
        "lng": area["center"][1],
        "summary": area["summary"]
    }
    
class FakeResponseLLM(SimpleChatModel):
    
    @property
    def _llm_type(self) -> str:
        return "fake-response-llm"
    
    def _call(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        payload = json.loads(
            messages[-1].content.split(
                "[위치 결과]\n",
                1
            )[1]
        )
        
        result, errors = payload["location_result"], payload["error_count"]
        
        if not result: 
            return (
                "죄송합니다. 요청하신 위치 정보를 찾지 못했습니다."
                "도로명 주소(예: '서울특별시 중구 세종대로 110')나 도시 이름을 포함해 다시 질문해 주세요."
            )
            
        if result["precision"] == "precise":
            retry_note = f" (일시적인 오류{errors}건이 있었지만 재시도로 복구했습니다.)" if errors else ""
            return (f"'{result['address']}'은(는) {result['name']}의 위치이며, "
                    f"좌표는 위도 {result['lat']}, 경도 {result['lng']} 입니다.{retry_note}")
            
        return (f"정확한 주소 위치는 확인하지 못해, 대신 '{result['city']}' 지역의 대략적인 정보를 알려드립니다. "
                f"중심 좌표는 위도 {result['lat']}, 경도 {result['lng']} 이며, {result['summary']}")

llm = FakeResponseLLM()

RESPONSE_ROLE = (
    "당신은 위치 안내 에이전트입니다. 아래 위치 결과를 사용자에게 명확하고 간결하게 전달하세요. "
    "precision 이 general 이면 정밀한 결과가 아니라는 점을 밝히세요. "
    "결과가 비어 있으면 사과하고, 사용자가 다음에 무엇을 하면 되는지 안내하세요."
)

class ErrorRecord(TypedDict):
    stage: str
    attempt: int
    error_type: str
    category: str
    message: str
    action: str 
    
class LocationAgentState(TypedDict):
    user_prompt: str
    location_result: dict
    primary_location_failed: bool 
    error_log: Annotated[
        list[ErrorRecord],
        operator.add 
    ]
    response: str 
    
def call_with_retry(tool, arg: str, stage: str, max_retries: int = MAX_RETRIES) -> tuple[dict | None, list[ErrorRecord]]:
    
    errors: list[ErrorRecord] = []
    for attempt in range(1, max_retries + 2):
        try:
            return tool(arg), errors
        
        except Exception as e:
            category = classify_error(e)
            can_retry = category == "transient" and attempt <= max_retries
            action = "재시도" if can_retry else ("포기 (재시도 소진)" if category == "transient" else "포기 (재시도 무의미)")
            errors.append({
                "stage": stage, 
                "attempt": attempt, 
                "error_type": type(e).__name__, 
                "category": category, 
                "message": str(e),
                "action": action
            })
            print(f"시도 {attempt}: {type(e).__name__} [{category}] -> {action}")
            
            if not can_retry:
                return None, errors 
            
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"{wait:.1f}초 대기 후 재시도(지수 백오프")
            time.sleep(wait)
            
    return None, errors

REQUEST_SUFFIX = re.compile(
    r"\s*(의)?\s*(위치|좌표|어디)\s*(\s+\S+)?\s*[??.!]*\s*$"
)

def extract_address(prompt: str) -> str:
    """서울특별시 중구 세종대로 110 위치 알려줘 -> 서울특별시 종구 세종대로 110"""
    address = REQUEST_SUFFIX.sub("", prompt.strip())
    
    if not address:
        raise ValueError("프롬프트에서 주소를 찾지 못했습니다.")
    
    return address 

def extract_city(prompt: str) -> str:
    """프롬프트의 첫 단어에서 도시 이름을 뽑는다. ('부산광역시 ...' -> '부산')"""
    
    words = prompt.split()
    
    if not words:
        raise ValueError("프롬프트가 비어 있습니다.")
    
    return re.sub(r"(특별자치도|특별자치시|특별시|광역시|시|도)$", "", words[0])

def primary_handler(state: LocationAgentState) -> dict:
    print("\n[E1] primary_handler: 정밀 위치 조회")
    
    try:
        address = extract_address(state["user_prompt"])
    except ValueError as e:
        print(f"입력 해석 실패: {e} -> 폴백")
        record: ErrorRecord = {
            "stage": "primary_handler",
            "attempt": 1, 
            "error_type": type(e).__name__,
            "category": "permanent",
            "massage": str(e),
            "action": "포기 (재시도 무의미)"
        }
        return {
            "primary_location_failed": True,
            "error_log": [record],
        }
        
    print(f" 주소 추출: '{address}'")
    
    result, errors = call_with_retry(
        get_precise_location_info,
        address, 
        stage = "primary_handler"
    )
    
    if result is None:
        return {
            "primary_location_failed": True,
            "error_log": errors
        }
        
    print("성공 {result['name']} ({result['lat']}, {result['lng']})")
    return {
        "primary_location_failed": False,
        "location_result": {
            "precision": "precise",
            **result
        },
        "error_log": errors
    }

def fallback_handler(state: LocationAgentState) -> dict:
    try:
        city = extract_city(state["user_prompt"])
        result = get_general_area_info(city)
    
    except Exception as e:
        print("플백 실패: {type(e).__name__}: {e}")
        record: ErrorRecord = {
            "stage": "fallback_handler",
            "attempt": 1, 
            "error_type": type(e).__name__,
            "category": classify_error(e),
            "message": str(e),
            "action": "포기 (응답 에이전트가 사과)"
        }
        return {
            "location_result": {},
            "error_log": [record]
        }
        
    print("폴백 성공: {city} 일대 ({result['lat']}, {result['lng']})")
    return {
        "location_result": {
            "precision": "general",
            **result
        }
    }
    
def response_agent(state: LocationAgentState) -> dict:
    print("\n[M2] response_agent: 답변 생성")
    payload = {
        "location_result": state.get("location_result") or {},
        "error_count": len(state["error_log"])
    }
    request = f"[사용자 질문]\n{state['user_prompt']}\n\n[위치 결과]\n{json.dumps(payload, ensure_ascii=False)}"
    response = llm.invoke([
        SystemMessage(content = RESPONSE_ROLE),
        HumanMessage(content = request)
    ])
    return {
        "response": response.content
    }
    
def route_after_primary(state: LocationAgentState) -> str:
    return "fallback" if state["primary_location_failed"] else "respond"


graph_builder = StateGraph(LocationAgentState)

graph_builder.add_node("primary_handler", primary_handler)
graph_builder.add_node("fallback_handler", fallback_handler)
graph_builder.add_node("response_agent", response_agent)

graph_builder.add_edge(START, "primary_handler")
graph_builder.add_conditional_edges(
    "primary_handler",
    route_after_primary,
    {
        "fallback": "fallback_handler",
        "respond": "response_agent"
    }
)
graph_builder.add_edge("fallback_handler", "response_agent")
graph_builder.add_edge("response_agent", END)

robust_locatioon_agent = graph_builder.compile()


# 실행 
DEMO_PROMPTS = [
    "서울특별시 중구 세종대로 110 위치 알려줘",
    "서울특별시 종로구 사직로 161 위치 알려줘",
    "부산광역시 해운대구 해운대해변로 999 어디야?",
    "제주특별자치도 제주시 첨단로 242 위치 알려줘",
    "화성 올림푸스 산 위치 알려줘",
]


def print_error_log(error_log: list[ErrorRecord]) -> None:
    if not error_log:
        print("[M3] 오류 기록: 없음")
        return
    print(f"[M3] 오류 기록: {len(error_log)}건")
    for r in error_log:
        print(f"    - {r['stage']} 시도 {r['attempt']} | {r['error_type']} [{r['category']}] | "
              f"{r['action']} | {r['message']}")


def handle_prompt(prompt: str) -> None:
    print(f"\n[M1] 사용자 프롬프트: {prompt}")
    reset_fault_simulation()
    try:
        final_state = robust_locatioon_agent.invoke(
            {"user_prompt": prompt, "location_result": {}, "primary_location_failed": False, "error_log": []}
        )
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
