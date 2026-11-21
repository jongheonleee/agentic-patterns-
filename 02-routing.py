"""
02. Routing 패턴

LLM(코디네이터)이 사용자 요청의 "의도"를 먼저 분류하고,
분류 결과에 따라 요청을 적절한 전문 핸들러(에이전트)로 분기시키는 패턴.

흐름:
    사용자 요청 → LLM 의도 분류 (booker / info / unclear)
      ├─ booker  → booking_handler (예약 처리)
      ├─ info    → info_handler    (정보 조회)
      └─ unclear → unclear_handler (명확화 요청, 기본 분기)
    → 처리 결과 출력

왜 쓰는가:
    - 하나의 거대한 프롬프트로 모든 경우를 처리하는 대신, 요청 유형별로 특화된 로직을 둘 수 있다.
    - 새 유형이 생기면 분류 기준과 핸들러만 추가하면 된다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
Fake 모델은 세 번의 분류 호출에 대해 "booker" → "info" → "unclear"를 순서대로 돌려준다.
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 라우터 지시문: LLM에게 "booker / info / unclear 중 하나만 출력"하도록 지시한다.
# 출력이 분기 조건으로 쓰이므로, 정해진 단어만 내도록 프롬프트를 엄격하게 작성하는 것이 핵심이다.
ROUTER_SYSTEM_PROMPT = """사용자의 요청을 분석하여 어떤 전문 핸들러가 처리해야 하는지 결정하세요.
            - 항공편이나 호텔 예약과 관련된 요청이면 'booker'를 출력하세요.
            - 일반적인 정보 질문이면 'info'를 출력하세요.
            - 요청이 불명확하거나 어느 카테고리에도 맞지 않으면 'unclear'를 출력하세요.
            'booker', 'info', 'unclear' 중 하나만 출력하세요.
        """

# 테스트 요청: Fake 모델의 응답 순서(booker → info → unclear)와 맞춘다.
REQUEST_A = "런던행 항공편 예약해줘."
REQUEST_B = "이탈리아의 수도가 어디야?"
REQUEST_C = "양자역학에 대해 알려줘."


# ---------------------------------------------------------------------------
# 2. Fake 모델 정의
#    분류(라우팅) 결과를 순서대로 돌려준다. main()의 테스트 순서와 일치시켜야 한다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    "booker",   # 요청 A: 항공편 예약 → 예약 핸들러
    "info",     # 요청 B: 수도 질문   → 정보 핸들러
    "unclear",  # 요청 C: 양자역학   → 불명확 처리
])


# ---------------------------------------------------------------------------
# 3. 전문 핸들러 정의
#    실제 시스템이라면 각각이 별도의 에이전트/체인/API 호출이 된다.
#    여기서는 어느 핸들러로 분기되었는지 보여주는 시뮬레이션 함수다.
# ---------------------------------------------------------------------------
def booking_handler(request: str) -> str:
    """예약 에이전트가 요청을 처리하는 동작을 시뮬레이션함"""
    print("\n--- 예약 핸들러로 위임 ---")
    return f"예약 핸들러 처리 완료: '{request}'. 결과: 예약 작업 시뮬레이션 완료."


def info_handler(request: str) -> str:
    """정보 에이전트가 요청을 처리하는 동작을 시뮬레이션함"""
    print("\n--- 정보 핸들러로 위임 ---")
    return f"정보 핸들러 처리 완료: '{request}'. 결과: 정보 조회 시뮬레이션 완료."


def unclear_handler(request: str) -> str:
    """위임할 수 없는 요청을 처리한다 (기본 분기)"""
    print("\n--- 불명확한 요청 처리 ---")
    return f"코디네이터가 요청을 위임할 수 없음: '{request}'. 요청을 명확히 해주세요."


# ---------------------------------------------------------------------------
# 4. 라우터(분류기) 체인
#    ROUTER_SYSTEM_PROMPT 로 요청을 booker / info / unclear 중 하나로 분류한다.
# ---------------------------------------------------------------------------
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("user", "{request}"),
])

coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 5. 분기별 처리 Runnable
#    RunnablePassthrough.assign(output=...)
#      → 들어온 입력 딕셔너리를 그대로 유지하면서 "output" 키만 새로 추가한다.
#    x['request']['request']인 이유:
#      → 6단계에서 "request": RunnablePassthrough()로 원본 입력({"request": "..."})을 통째로 담았기 때문에
#        x = {"decision": "...", "request": {"request": "실제 문장"}} 구조가 된다.
# ---------------------------------------------------------------------------
branches = {
    "booker": RunnablePassthrough.assign(
        output=lambda x: booking_handler(x['request']['request'])
    ),
    "info": RunnablePassthrough.assign(
        output=lambda x: info_handler(x['request']['request'])
    ),
    "unclear": RunnablePassthrough.assign(
        output=lambda x: unclear_handler(x['request']['request'])
    ),
}

# RunnableBranch: (조건, 실행할 Runnable) 쌍을 위에서부터 검사하고, 모두 거짓이면 마지막 기본값을 실행한다.
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip() == 'info', branches["info"]),
    branches["unclear"],   # 기본 분기 (조건 없음)
)


# ---------------------------------------------------------------------------
# 6. 코디네이터 에이전트 조립
#    딕셔너리 Runnable은 각 키를 "병렬"로 계산한다.
#      "decision": LLM이 분류한 결과 ("booker" 등)
#      "request" : 원본 입력을 그대로 보존 (핸들러가 실제 문장을 써야 하므로)
#    → delegation_branch가 decision을 보고 분기
#    → 마지막 람다가 핸들러 결과(output)만 꺼내 최종 반환
# ---------------------------------------------------------------------------
coordinator_input = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough(),
}

coordinator_agent = coordinator_input | delegation_branch | (lambda x: x['output'])


# ---------------------------------------------------------------------------
# 7. 실행
#    Fake 모델의 응답 순서(booker → info → unclear)와 테스트 순서를 맞춘다.
# ---------------------------------------------------------------------------
def main() -> None:
    print("--- 예약 요청 테스트 ---")
    result_a = coordinator_agent.invoke({"request": REQUEST_A})
    print(f"최종 결과 A: {result_a}")

    print("\n--- 정보 요청 테스트 ---")
    result_b = coordinator_agent.invoke({"request": REQUEST_B})
    print(f"최종 결과 B: {result_b}")

    print("\n--- 불명확한 요청 테스트 ---")
    result_c = coordinator_agent.invoke({"request": REQUEST_C})
    print(f"최종 결과 C: {result_c}")


if __name__ == "__main__":
    main()
