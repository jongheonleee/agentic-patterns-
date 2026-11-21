"""
03. Parallelization 패턴

서로 독립적인 여러 LLM 작업을 "동시에" 실행한 뒤,
그 결과들을 모아 하나의 종합 답변을 만드는 패턴. (Map → Reduce 구조)

흐름:
    주제 입력
      → [병렬] 요약 생성 / 질문 생성 / 핵심 용어 추출   (RunnableParallel)
      → 결과 취합 {summary, questions, key_terms, topic}
      → 종합 답변 생성
      → 최종 답변 출력

왜 쓰는가:
    - 작업 간 의존성이 없으면 순차 실행할 이유가 없다. 병렬로 돌리면 전체 지연 시간이 가장 느린 작업 하나로 줄어든다.
    - 서로 다른 관점(요약/질문/용어)을 독립적으로 뽑은 뒤 합치면 더 풍부한 결과를 얻을 수 있다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
주의: FakeListChatModel은 응답을 "호출 순서대로" 돌려주므로, 병렬 실행 시 어떤 체인이 어떤 응답을 받을지는
      실행 순서에 따라 달라질 수 있다. 실제 모델에서는 각 체인이 자기 프롬프트에 맞는 응답을 받는다.
"""
import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 입력 주제
TEST_TOPIC = "우주 탐사의 역사"

# 병렬 체인 3개의 지시문 (세 체인 모두 같은 입력 {topic} 을 받는다)
SUMMARIZE_SYSTEM_PROMPT = "다음 주제를 간결하게 요약하세요:"
QUESTIONS_SYSTEM_PROMPT = "다음 주제에 대해 심화되는 3가지 질문을 생성하세요:"
TERMS_SYSTEM_PROMPT = "다음 주제에서 5-10개의 핵심 용어를 쉼표로 구분하여 추출하세요:"

# 종합(Reduce) 지시문: map_chain 의 출력 키({summary}, {questions}, {key_terms})가 입력 변수로 들어간다.
SYNTHESIS_SYSTEM_PROMPT = """
        다음 정보를 바탕으로 종합적인 답변을 작성하세요:
        요약: {summary}
        관련 질문: {questions}
        핵심 용어: {key_terms}
    """


# ---------------------------------------------------------------------------
# 2. Fake 모델 정의
#    병렬 체인 3개 + 종합 체인 1개 = 총 4번 호출되므로 응답 4개를 준비한다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # 병렬 1: 요약
    (
        "우주 탐사는 1957년 스푸트니크 발사로 시작되어 달 착륙, 우주정거장 건설 등을 거쳐 "
        "현재 화성 탐사를 목표로 하고 있다."
    ),
    # 병렬 2: 심화 질문
    "1. 우주 탐사의 주요 이정표는 무엇인가?\n2. 민간 우주 기업의 역할은?\n3. 미래 우주 탐사의 방향은?",
    # 병렬 3: 핵심 용어
    "스푸트니크, 아폴로, NASA, SpaceX, 국제우주정거장, 화성, 로켓, 위성",
    # 종합: 위 결과를 바탕으로 한 최종 답변
    (
        "우주 탐사는 1957년 소련의 스푸트니크 발사로 시작되었습니다. "
        "이후 미국의 아폴로 프로그램으로 인류가 달에 착륙했고, 국제우주정거장이 건설되었습니다. "
        "현재는 SpaceX 등 민간 기업이 참여하며 화성 탐사를 목표로 발전하고 있습니다."
    ),
])


# ---------------------------------------------------------------------------
# 3. 병렬로 실행할 독립 체인 3개
#    세 체인 모두 같은 입력({topic})을 받지만, 서로의 결과에 의존하지 않는다.
# ---------------------------------------------------------------------------
# 체인 A: 주제 요약
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", SUMMARIZE_SYSTEM_PROMPT),
        ("user", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)

# 체인 B: 심화 질문 생성
questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", QUESTIONS_SYSTEM_PROMPT),
        ("user", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)

# 체인 C: 핵심 용어 추출
terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", TERMS_SYSTEM_PROMPT),
        ("user", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------------------------
# 4. 병렬 실행 (Map 단계)
#    RunnableParallel은 같은 입력을 모든 체인에 동시에 넘기고,
#    결과를 {키: 출력} 딕셔너리로 모아준다.
#    "topic": RunnablePassthrough() → 원본 입력도 함께 보존해 다음 단계에서 쓸 수 있게 한다.
# ---------------------------------------------------------------------------
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),
    }
)


# ---------------------------------------------------------------------------
# 5. 종합 (Reduce 단계)
#    map_chain의 출력 딕셔너리 키({summary}, {questions}, {key_terms}, {topic})가
#    그대로 이 프롬프트의 입력 변수로 들어간다. 키 이름이 일치해야 한다.
# ---------------------------------------------------------------------------
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIS_SYSTEM_PROMPT),
    ("user", "원본 주제: {topic}"),
])

# 전체 체인: 병렬 처리 → 종합 프롬프트 → LLM → 문자열
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 6. 실행
#    ainvoke()를 쓰면 RunnableParallel 내부의 체인들이 실제로 비동기 병렬로 실행된다.
#    (동기 invoke()도 동작하지만 스레드 풀 기반이다.)
# ---------------------------------------------------------------------------
async def run_parallel_example(topic: str) -> None:
    print(f"\n--- 병렬 처리 예제 실행: '{topic}' ---")

    try:
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- 최종 응답 ---")
        print(response)
    except Exception as e:
        print(f"\n체인 실행 중 오류 발생: {e}")


async def main() -> None:
    await run_parallel_example(TEST_TOPIC)


if __name__ == "__main__":
    asyncio.run(main())
