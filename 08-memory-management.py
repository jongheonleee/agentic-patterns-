"""
08. Memory Management 패턴

에이전트가 "지금 대화"의 문맥(단기 기억)과 "여러 세션에 걸친 지식"(장기 기억)을
저장·검색·갱신하여 맥락을 유지하고, 사용자에 맞춰 행동을 점점 개선하는 패턴.

기억의 종류:
    - 단기 기억 (Short-term)  : 한 대화(thread) 안의 메시지 이력. 체크포인터가 thread_id 단위로 저장한다.
    - 장기 기억 (Long-term)   : 세션이 끝나도 남는 사용자 정보(이름, 선호 등). 스토어에 네임스페이스로 저장한다.
    - 절차적 기억 (Procedural): 에이전트가 "어떻게 행동할지"를 담은 지침. 회고(reflection)로 스스로 갱신한다.

흐름:
    사용자 메시지 입력 (thread_id, user_id)
      → [기억 로드]  L1. 단기: 체크포인터가 thread_id의 대화 이력을 복원
                     L2. 장기: 스토어에서 (user_id, "memories") 네임스페이스를 의미 검색
                     L3. 절차: 스토어에서 에이전트 지침을 조회
      → 모델 호출 (지침 + 장기 기억 + 대화 이력 → 응답)
      → [기억 저장]  S1. 이번 턴에서 새로 알게 된 사실 추출
                     S2. 있으면 스토어에 저장 / 없으면(NONE) 생략
      → 응답 출력 (체크포인터가 대화 상태를 자동 저장)
      ↻ 다음 메시지(같은 thread_id)는 이력을 이어서 처리
      → 세션 종료 후 [회고] P1. 대화를 돌아보며  P2. 지침 갱신 → 다음 세션에 적용

왜 쓰는가:
    - LLM은 본질적으로 상태가 없다(stateless). 기억 장치가 없으면 바로 앞 문장도 잊는다.
    - 단기 기억만 있으면 새 세션마다 사용자가 같은 설명을 반복해야 한다. 장기 기억이 이를 해결한다.
    - 절차적 기억은 "경험에서 배운 행동 규칙"을 축적해 에이전트를 점점 더 쓸모 있게 만든다.

LangGraph 사용 포인트:
    - MessagesState: add_messages 리듀서가 메시지를 "덮어쓰기"가 아닌 "누적"으로 합친다. (= 단기 기억의 실체)
    - checkpointer(InMemorySaver): 노드 실행이 끝날 때마다 상태를 thread_id 키로 저장한다.
    - store(InMemoryStore): 그래프 밖에서도 살아 있는 키-값 저장소. (namespace, key)로 구분하고 의미 검색을 지원한다.
    - 노드 함수에 `store: BaseStore` 매개변수를 선언하면, compile() 시 넘긴 스토어를 LangGraph가 주입한다.

LangChain 0.x의 ConversationBufferMemory / LLMChain 은 1.0부터 `langchain` 패키지에서 빠져
레거시 패키지(langchain-classic)로 분리되었다. 그 역할은 LangGraph의 체크포인터(단기 기억)와 스토어(장기 기억)가 대신한다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
- llm        : 사용자에게 답하는 대화 모델 (세션 1에서 4번, 세션 2·3에서 각 1번 호출)
- memory_llm : 기억을 관리하는 모델 (턴마다 "새 사실 추출" 1번, 세션 1 종료 후 "지침 회고" 1번 호출)
"""
import uuid
import zlib

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 가짜 임베딩 벡터의 차원 수
EMBED_DIMS = 128

# 절차적 기억(에이전트 지침)은 사용자와 무관하므로 별도 네임스페이스에 하나의 키로 둔다.
INSTRUCTIONS_NAMESPACE = ("agent_instructions",)
INSTRUCTIONS_KEY = "travel_agent"
INITIAL_INSTRUCTIONS = "당신은 친절한 여행 비서입니다. 항공편·호텔·일정 관련 요청을 돕습니다."

# 기억 관리 모델에게 주는 역할
EXTRACT_ROLE = (
    "당신은 기억 관리자입니다. 아래 대화 한 턴에서 앞으로도 기억할 가치가 있는 사용자 정보"
    "(이름, 선호, 일정, 제약 조건 등)를 한 문장으로 추출하세요. 없으면 'NONE'만 출력하세요."
)

REFLECT_ROLE = (
    "당신은 여행 비서 에이전트의 코치입니다. 현재 지침과 대화 기록을 보고, "
    "다음 세션에서 더 잘 응대하기 위해 지침에 추가할 행동 규칙을 도출하세요. "
    "기존 지침을 유지하면서 규칙을 글머리 기호로 덧붙인 '새 지침 전체'를 출력하세요."
)

# 데모 입력
USER_ID = "sam"

SESSION_1_INPUTS = [
    "다음 주에 뉴욕으로 출장을 가요.",
    "항공편을 예약하고 싶어요. 저는 창가 좌석을 좋아해요.",
    "그리고 제 이름은 샘이에요.",
    "제 이름이 뭐였죠?",
]
SESSION_2_INPUT = "안녕하세요, 이번엔 도쿄 여행을 계획 중이에요."
SESSION_3_INPUT = "다음 달에는 파리 출장도 있어요."


# ---------------------------------------------------------------------------
# 2. 상태(State) 정의
#    MessagesState의 `messages` 키는 add_messages 리듀서로 누적된다. (단기 기억)
#    여기에 "이번 턴에 스토어에서 꺼내 온 것"과 "이번 턴에 새로 알게 된 것"을 담는 키를 추가한다.
# ---------------------------------------------------------------------------
class MemoryAgentState(MessagesState):
    recalled_memories: list[str]   # L2: 스토어에서 검색한 장기 기억
    instructions: str              # L3: 스토어에서 읽은 절차적 기억(지침)
    new_fact: str                  # S1: 이번 턴에서 추출한 새 사실 ("NONE"이면 없음)


# ---------------------------------------------------------------------------
# 3. Fake 모델 정의
#    대화 모델과 기억 관리 모델을 분리해 두면, 각 모델의 호출 순서를 따로 추적할 수 있다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # [세션 1] 턴 1~4: 같은 thread_id → 단기 기억(대화 이력)이 누적된다
    "뉴욕 출장이시군요! 항공편, 호텔, 현지 교통 중 무엇부터 도와드릴까요?",
    "뉴욕행 항공편을 창가 좌석 기준으로 찾아보겠습니다. 출발 날짜와 선호 시간대를 알려주시면 바로 조회할게요.",
    "반갑습니다, 샘! 예약자 이름은 샘으로 기록해 두겠습니다.",
    "샘이라고 알려주셨어요. 뉴욕행 창가 좌석 항공편 조회를 계속 진행할까요?",

    # [세션 2] 새 thread_id → 단기 기억은 비어 있지만, 장기 기억(스토어)으로 이름·선호를 기억한다
    "다시 만나 반갑습니다, 샘! 이번엔 도쿄 여행이군요. 지난번처럼 창가 좌석으로 항공편을 찾아볼까요?",

    # [세션 3] 회고로 갱신된 지침(절차적 기억)이 적용된 응답
    "샘, 파리 출장 준비도 함께할게요. 항공편은 평소 선호하시는 창가 좌석으로 먼저 찾아보고, "
    "호텔은 회의 장소 근처로 추천드릴게요.",
])

memory_llm = FakeListChatModel(responses=[
    # [세션 1] 턴마다 새 사실 추출 (없으면 NONE)
    "사용자는 다음 주 뉴욕 출장을 앞두고 있다.",
    "사용자는 비행기에서 창가 좌석을 선호한다.",
    "사용자의 이름은 샘이다.",
    "NONE",

    # [세션 2] 새 사실 추출
    "사용자는 도쿄 여행을 계획 중이다.",

    # [회고] 세션 1 대화를 돌아보며 지침 갱신
    """당신은 친절한 여행 비서입니다. 항공편·호텔·일정 관련 요청을 돕습니다.
- 사용자의 이름을 알고 있으면 이름을 불러 인사합니다.
- 좌석 선호(창가/통로)를 기억해 항공편 조회 시 먼저 적용합니다.
- 출장인 경우 회의 장소 근처 호텔을 함께 제안합니다.""",

    # [세션 3] 새 사실 추출
    "사용자는 다음 달 파리 출장 예정이다.",
])


def fake_embed(texts: list[str]) -> list[list[float]]:
    """학습용 가짜 임베딩: 글자 2개 단위(바이그램)를 해시해 고정 차원 벡터에 담는다.
    실제로는 OpenAIEmbeddings 같은 임베딩 모델을 넣는다. 겹치는 단어가 많을수록 코사인 유사도가 높아진다."""
    vectors = []
    for text in texts:
        vec = [0.0] * EMBED_DIMS
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            vec[zlib.crc32(bigram.encode("utf-8")) % EMBED_DIMS] += 1.0
        vectors.append(vec)
    return vectors


# ---------------------------------------------------------------------------
# 4. 헬퍼 함수
# ---------------------------------------------------------------------------
def memories_namespace(config: RunnableConfig) -> tuple[str, str]:
    """장기 기억은 사용자별로 분리한다: (user_id, "memories")"""
    return (config["configurable"]["user_id"], "memories")


def build_system_prompt(instructions: str, memories: list[str]) -> str:
    """절차적 기억(지침)과 장기 기억을 시스템 프롬프트 하나로 합친다."""
    prompt = instructions
    if memories:
        prompt += "\n\n[사용자에 대해 기억하는 것]\n" + "\n".join(f"- {m}" for m in memories)
    return prompt


# ---------------------------------------------------------------------------
# 5. 노드 정의
# ---------------------------------------------------------------------------
def load_memories(
    state: MemoryAgentState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    """[L1~L3] 세 종류의 기억을 불러온다.
    L1(단기)은 체크포인터가 이미 state["messages"]에 복원해 두었으므로, 여기서는 확인만 한다."""
    print(f"--- L1. 단기 기억: 이 thread의 메시지 {len(state['messages'])}개 복원됨 ---")

    # L2. 장기 기억: 최신 사용자 메시지를 질의로 삼아 관련 기억을 의미 검색한다.
    latest_user_message = state["messages"][-1].content
    items = store.search(
        memories_namespace(config),
        query=latest_user_message,
        limit=5,
    )
    memories = [item.value["content"] for item in items]
    print(f"--- L2. 장기 기억 검색 결과 ({len(memories)}개) ---")
    for item in items:
        print(f"    · {item.value['content']}  (유사도 {item.score:.2f})")

    # L3. 절차적 기억: 현재 에이전트 지침을 읽는다.
    instructions = store.get(INSTRUCTIONS_NAMESPACE, INSTRUCTIONS_KEY).value["instructions"]

    return {"recalled_memories": memories, "instructions": instructions}


def call_model(state: MemoryAgentState) -> dict:
    """[M2] 지침 + 장기 기억(시스템 프롬프트) + 대화 이력(단기 기억)으로 모델을 호출한다."""
    system_prompt = build_system_prompt(state["instructions"], state["recalled_memories"])
    print("--- M2. 모델 호출 (시스템 프롬프트) ---")
    print("    " + system_prompt.replace("\n", "\n    "))

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])
    return {"messages": [response]}   # add_messages 리듀서가 기존 이력 뒤에 덧붙인다


def extract_memory(state: MemoryAgentState) -> dict:
    """[S1] 이번 턴(사용자 메시지 + AI 응답)에서 장기 기억으로 남길 사실을 추출한다."""
    user_message = state["messages"][-2]
    ai_message = state["messages"][-1]
    turn = f"사용자: {user_message.content}\nAI: {ai_message.content}"

    response = memory_llm.invoke([
        SystemMessage(content=EXTRACT_ROLE),
        HumanMessage(content=turn),
    ])
    fact = response.content.strip()
    return {"new_fact": fact}


def has_new_fact(state: MemoryAgentState) -> str:
    """[S1 → S2 분기] 저장할 사실이 있으면 save, 없으면 skip"""
    if state["new_fact"] == "NONE":
        print("--- S1. 새로 기억할 사실 없음 (저장 생략) ---")
        return "skip"
    return "save"


def save_memory(
    state: MemoryAgentState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> None:
    """[S2] 추출한 사실을 사용자 네임스페이스에 저장한다. 키는 고유 ID로 둔다."""
    store.put(
        memories_namespace(config),
        str(uuid.uuid4()),
        {"content": state["new_fact"]},
    )
    print(f"--- S2. 장기 기억 저장: '{state['new_fact']}' ---")


# ---------------------------------------------------------------------------
# 6. 절차적 기억 갱신 (회고)
#    세션이 끝난 뒤 대화를 돌아보며 "다음부터 이렇게 행동하자"는 지침을 다시 쓴다.
#    04. Reflection 패턴을 기억 갱신에 적용한 것으로, 그래프 밖에서 실행한다.
# ---------------------------------------------------------------------------
def update_instructions(conversation: list[BaseMessage], store: BaseStore) -> None:
    """[P1~P2] 대화를 회고하고(P1) 스토어의 지침을 갱신한다(P2)."""
    current = store.get(INSTRUCTIONS_NAMESPACE, INSTRUCTIONS_KEY).value["instructions"]
    transcript = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in conversation
    )

    print("--- P1. 대화 회고 중... ---")
    response = memory_llm.invoke([
        SystemMessage(content=REFLECT_ROLE),
        HumanMessage(content=f"[현재 지침]\n{current}\n\n[대화 기록]\n{transcript}"),
    ])
    new_instructions = response.content

    store.put(INSTRUCTIONS_NAMESPACE, INSTRUCTIONS_KEY, {"instructions": new_instructions})
    print("--- P2. 지침 갱신 완료 ---")
    print("    " + new_instructions.replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# 7. 그래프 구성
#    InMemoryStore는 (namespace, key) → value 구조의 저장소다.
#    index에 임베딩 함수를 넣으면 store.search(query=...)로 "의미 검색"이 가능해진다.
#
#    START → load_memories → call_model → extract_memory ─(save)→ save_memory → END
#                                                        └(skip)──────────────→ END
#    compile() 에 checkpointer(단기 기억)와 store(장기 기억)를 함께 넘긴다.
# ---------------------------------------------------------------------------
store = InMemoryStore(index={"embed": fake_embed, "dims": EMBED_DIMS})
store.put(
    INSTRUCTIONS_NAMESPACE,
    INSTRUCTIONS_KEY,
    {"instructions": INITIAL_INSTRUCTIONS},
)

graph_builder = StateGraph(MemoryAgentState)

graph_builder.add_node("load_memories", load_memories)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("extract_memory", extract_memory)
graph_builder.add_node("save_memory", save_memory)

graph_builder.add_edge(START, "load_memories")
graph_builder.add_edge("load_memories", "call_model")
graph_builder.add_edge("call_model", "extract_memory")
graph_builder.add_conditional_edges(
    "extract_memory",
    has_new_fact,
    {
        "save": "save_memory",
        "skip": END,
    },
)
graph_builder.add_edge("save_memory", END)

memory_agent = graph_builder.compile(
    checkpointer=InMemorySaver(),   # 단기 기억: thread_id 단위로 상태 저장
    store=store,                    # 장기 기억: 노드의 `store` 매개변수로 주입
)


# ---------------------------------------------------------------------------
# 8. 실행
# ---------------------------------------------------------------------------
def chat(thread_id: str, user_id: str, text: str) -> None:
    """한 턴 대화. thread_id는 단기 기억(대화)의 키, user_id는 장기 기억(사용자)의 키다."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    print(f"\n[사용자] {text}")

    result = memory_agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        config=config,
    )
    print(f"[AI] {result['messages'][-1].content}")


def main() -> None:
    print("=" * 25 + " 세션 1: thread 'trip-1' (단기 기억 누적) " + "=" * 25)
    for text in SESSION_1_INPUTS:
        chat("trip-1", USER_ID, text)

    print("\n\n" + "=" * 25 + " 세션 2: 새 thread 'trip-2' (장기 기억으로 회상) " + "=" * 25)
    chat("trip-2", USER_ID, SESSION_2_INPUT)

    print("\n\n" + "=" * 25 + " 회고: 세션 1을 돌아보며 지침 갱신 (절차적 기억) " + "=" * 25)
    # 체크포인터에서 세션 1의 대화 이력을 꺼내 회고 재료로 쓴다.
    session_1 = memory_agent.get_state({"configurable": {"thread_id": "trip-1"}})
    update_instructions(session_1.values["messages"], store)

    print("\n\n" + "=" * 25 + " 세션 3: 새 thread 'trip-3' (갱신된 지침 적용) " + "=" * 25)
    chat("trip-3", USER_ID, SESSION_3_INPUT)

    print("\n\n" + "=" * 30 + " 스토어에 남은 장기 기억 " + "=" * 30)
    for item in store.search((USER_ID, "memories")):
        print(f"- {item.value['content']}")


if __name__ == "__main__":
    main()
