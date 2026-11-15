"""
05. Function Calling (Tool Calling) 패턴

LLM이 직접 답하는 대신 "어떤 도구를 어떤 인자로 호출할지"를 결정하고,
프레임워크가 그 도구를 실행한 뒤 결과를 다시 LLM에 전달하여 최종 답변을 만드는 패턴.

흐름:
    사용자 질문 → LLM 호출(도구 목록 바인딩)
      ├─ tool_calls 있음 → 도구 실행 → ToolMessage 전달 → LLM 재호출 → 최종 답변
      └─ tool_calls 없음 → 바로 최종 답변

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
Fake 모델은 "도구 호출 요청(AIMessage + tool_calls)" → "최종 답변(AIMessage)" 순서로
미리 정해둔 응답을 돌려준다.
"""
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# 1. 도구 정의
#    @tool 데코레이터가 함수 시그니처와 docstring을 읽어 LLM에 전달할 "도구 스키마"를 만든다.
#    docstring은 LLM이 도구의 용도를 판단하는 설명이므로 반드시 작성한다.
# ---------------------------------------------------------------------------
@tool
def search_information(query: str) -> str:
    """주어진 검색어로 정보를 검색한다. 사실 확인이나 최신 정보가 필요할 때 사용한다."""
    print(f"\n--- 도구 호출: search_information(query='{query}') ---")

    # 실제 검색 API 대신 시뮬레이션 결과를 돌려준다.
    simulated_results = {
        "런던 날씨": "현재 런던은 흐리고 기온은 15°C입니다.",
        "프랑스 수도": "프랑스의 수도는 파리입니다.",
        "지구 인구": "지구의 인구는 약 80억 명으로 추정됩니다.",
        "가장 높은 산": "에베레스트산은 해발 기준으로 세계에서 가장 높은 산입니다.",
    }

    # 검색어에 키워드가 포함되어 있으면 해당 결과를 반환 (부분 일치)
    for keyword, answer in simulated_results.items():
        if keyword in query:
            result = answer
            break
    else:
        result = f"'{query}'에 대한 시뮬레이션 검색 결과: 구체적인 정보는 찾지 못했지만 흥미로운 주제로 보입니다."

    print(f"--- 도구 결과: {result} ---")
    return result


tools = [search_information]


# ---------------------------------------------------------------------------
# 2. Fake 모델 정의
#    GenericFakeChatModel은 AIMessage를 순서대로 돌려주므로 tool_calls를 포함한 응답을 흉내낼 수 있다.
#    다만 bind_tools()가 구현되어 있지 않아, 에이전트가 도구를 바인딩할 수 있도록 오버라이드한다.
# ---------------------------------------------------------------------------
class FakeToolCallingModel(GenericFakeChatModel):
    """도구 호출(tool_calls) 응답을 흉내내는 학습용 Fake 모델"""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        # 실제 모델은 여기서 도구 스키마를 요청에 포함시킨다.
        # Fake 모델은 미리 정해둔 응답을 돌려줄 뿐이므로 자기 자신을 그대로 반환한다.
        return self


def make_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """LLM이 '이 도구를 이 인자로 호출해 달라'고 요청하는 AIMessage를 만든다."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# 질문 3개 × (도구 호출 요청 → 최종 답변) = 응답 6개. 호출 순서대로 소비된다.
llm = FakeToolCallingModel(messages=iter([
    # Q1. 프랑스의 수도
    make_tool_call("search_information", {"query": "프랑스 수도"}, "call_1"),
    AIMessage(content="프랑스의 수도는 파리입니다."),

    # Q2. 런던 날씨
    make_tool_call("search_information", {"query": "런던 날씨"}, "call_2"),
    AIMessage(content="현재 런던은 흐린 날씨이며 기온은 15°C입니다."),

    # Q3. 강아지 (도구가 구체적인 정보를 찾지 못하는 경우)
    make_tool_call("search_information", {"query": "강아지"}, "call_3"),
    AIMessage(content="검색에서 구체적인 정보를 찾지 못했지만, 강아지는 인류와 가장 오래 함께한 반려동물 중 하나입니다."),
]))


# ---------------------------------------------------------------------------
# 3. 에이전트 생성
#    create_agent가 내부적으로 "LLM 호출 → tool_calls 있으면 도구 실행 → 결과를 LLM에 재전달" 루프를 수행한다.
#    (LangChain 1.0부터 AgentExecutor / create_tool_calling_agent 대신 create_agent를 사용한다.)
# ---------------------------------------------------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="당신은 도움이 되는 어시스턴트입니다. 사실 확인이 필요하면 search_information 도구를 사용하세요.",
)


def run_agent_with_tool(query: str) -> None:
    print("\n" + "=" * 20 + f" 질문: '{query}' " + "=" * 20)

    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    # 에이전트가 주고받은 메시지 흐름을 출력해 Function Calling 과정을 확인한다.
    print("\n--- 메시지 흐름 ---")
    for message in response["messages"]:
        role = type(message).__name__
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[{role}] 도구 호출 요청 → {call['name']}({call['args']})")
        else:
            print(f"[{role}] {message.content}")

    print("\n--- 최종 답변 ---")
    print(response["messages"][-1].content)


def main() -> None:
    queries = [
        "프랑스의 수도는 어디인가요?",
        "런던 날씨는 어때?",
        "강아지에 대해 알려줘",
    ]
    # Fake 모델은 응답을 순서대로 소비하므로 질문을 순차적으로 처리한다.
    # (실제 모델이라면 asyncio.gather + agent.ainvoke로 병렬 실행도 가능하다.)
    for query in queries:
        run_agent_with_tool(query)


if __name__ == "__main__":
    main()
