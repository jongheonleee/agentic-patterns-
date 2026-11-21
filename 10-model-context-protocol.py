"""
10. Model Context Protocol (MCP) 패턴

MCP는 LLM 애플리케이션(호스트/클라이언트)과 외부 시스템(서버)을 잇는 표준 프로토콜이다.
"AI를 위한 USB-C"에 비유되는데, 서버가 한 번 MCP 규격으로 기능을 노출하면
어떤 클라이언트(Claude Desktop, IDE, LangChain 에이전트 …)든 같은 방식으로 꽂아 쓸 수 있다.

MCP 서버가 노출하는 세 가지 기본 요소(primitives):
    - Tools     : 모델이 "호출"하는 동작  (예: DB 조회, 웹 검색, HTTP 요청)   → tools/list, tools/call
    - Resources : 모델이 "읽는" 데이터     (예: DB 스키마, 상품 상세, 설정 파일) → resources/list, resources/read
    - Prompts   : 재사용 가능한 프롬프트 템플릿                                → prompts/list, prompts/get

통신 방식(transport): 메시지는 JSON-RPC 2.0 이고, 전송은 stdio(로컬 프로세스) 또는 HTTP(원격)를 쓴다.

흐름:
    사용자 질문 입력
      → [MCP 클라이언트]  C1. 서버 연결 · 기능 조회 (initialize → tools/list …)
                          C2. LLM 호출 (MCP 도구 스키마 바인딩) → tool_calls 결정
                          C3. tools/call 요청 전송 (JSON-RPC over stdio)
      → [MCP 서버]        S1. 요청 라우팅 → S2. 도구 실행 / 리소스 읽기 / 프롬프트 생성 → S3. 결과 직렬화·응답
      ↻ 결과를 ToolMessage로 LLM에 재전달 (tool_calls가 없을 때까지 반복)
      → 최종 답변 출력

05. Function Calling 과의 차이:
    - Function Calling 은 도구가 에이전트 코드 "안"에 파이썬 함수로 있다.
    - MCP 는 도구가 "별도 프로세스/서버"에 있고, 에이전트는 실행 시점에 목록을 조회해 동적으로 연결한다.
      따라서 도구를 여러 에이전트·여러 언어·여러 제품이 공유할 수 있다.

이 파일 하나가 서버와 클라이언트 역할을 모두 한다:
    python 10-model-context-protocol.py            # 데모 실행 (클라이언트 → 이 파일을 서버 프로세스로 띄움)
    python 10-model-context-protocol.py --server   # MCP 서버만 stdio 로 실행 (Claude Desktop 등에 등록할 때)
    python 10-model-context-protocol.py --http     # MCP 서버를 HTTP(127.0.0.1:8000/mcp) 로 실행

학습용이므로 LLM은 실제 API 대신 Fake 모델(05와 같은 FakeToolCallingModel)을 사용한다.
반면 MCP 서버·프로토콜·SQLite 조회는 전부 실제로 동작한다. (웹 검색만 시뮬레이션, HTTP 요청은 실제)
"""
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import fastmcp
import httpx
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

fastmcp.settings.log_level = "WARNING"   # 학습용 출력에 서버 INFO 로그가 섞이지 않도록


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# MCP 서버 정보
SERVER_NAME = "shop-assistant"
SERVER_INSTRUCTIONS = "쇼핑몰 운영 데이터(SQLite)와 웹 정보를 제공하는 MCP 서버입니다."

# HTTP 모드(--http)로 실행할 때의 주소
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000

# 도구 · 리소스 동작 설정
MAX_QUERY_ROWS = 20           # query_database 가 돌려줄 최대 행 수
FETCH_TIMEOUT_SECONDS = 5.0   # fetch_url 의 HTTP 요청 타임아웃
FETCH_PREVIEW_CHARS = 500     # fetch_url 이 돌려줄 본문 앞부분 길이
README_PREVIEW_CHARS = 300    # docs://readme 리소스가 돌려줄 README 앞부분 길이

# 샘플 데이터베이스 (테이블 구조 + 데이터)
DB_SEED_SQL = """
    CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price INTEGER);
    CREATE TABLE orders   (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER, ordered_at TEXT);

    INSERT INTO products VALUES
        (1, '무선 이어폰 Air',   '전자기기', 89000),
        (2, '요가 매트 프로',    '스포츠',   35000),
        (3, '스테인리스 텀블러', '생활',     22000),
        (4, '블루투스 키보드',   '전자기기', 64000);

    INSERT INTO orders VALUES
        (1, 1, 12, '2026-10-03'), (2, 2, 8,  '2026-10-05'), (3, 1, 15, '2026-10-11'),
        (4, 3, 20, '2026-10-14'), (5, 4, 6,  '2026-10-18'), (6, 1, 15, '2026-10-25'),
        (7, 3, 9,  '2026-10-29'), (8, 2, 11, '2026-11-02');
"""

# 시뮬레이션용 웹 검색 결과 (검색어에 포함된 키워드 → 결과 목록)
SIMULATED_WEB = {
    "무선 이어폰 Air": [
        {
            "title": "무선 이어폰 Air 한 달 사용기 — 가성비는 확실",
            "url": "https://example.com/review/air",
            "snippet": "노이즈 캔슬링은 무난하고 배터리는 7시간. 가격 대비 만족도가 높다는 평이 많다.",
        },
        {
            "title": "[비교] 10만원대 무선 이어폰 5종",
            "url": "https://example.com/compare/earbuds",
            "snippet": "Air 모델은 착용감 부문 1위, 통화 품질은 중위권.",
        },
    ],
    "요가 매트 프로": [
        {
            "title": "요가 매트 프로 두께 8mm 실사용 후기",
            "url": "https://example.com/review/yoga",
            "snippet": "미끄럼 방지가 좋고 냄새가 적다. 보관 스트랩이 포함되어 편리.",
        },
    ],
}

# 클라이언트 A(서버 기능 탐색)에서 사용할 예시 입력
DEMO_SQL = "SELECT category, SUM(price) AS total FROM products GROUP BY category"
FORBIDDEN_SQL = "DELETE FROM orders"
BRIEFING_MONTH = "2026년 10월"

# 클라이언트 B(LangChain 에이전트)의 역할과 질문
AGENT_SYSTEM_PROMPT = (
    "당신은 쇼핑몰 운영 어시스턴트입니다. "
    "판매 데이터는 query_database, 외부 평판은 search_web 으로 확인하세요."
)
USER_QUESTION = "지난달(2026년 10월) 가장 많이 팔린 상품이 뭐야? 그 상품의 온라인 반응도 알려줘."


# ---------------------------------------------------------------------------
# 2. [서버 측] FastMCP 서버 정의
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name=SERVER_NAME,
    instructions=SERVER_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# 3. [서버 측] 데이터베이스 연동 (SQLite)
#    실제 서비스라면 PostgreSQL/MySQL 커넥션 풀을 두는 자리다. 여기서는 메모리 DB에 샘플 데이터를 넣는다.
#    FastMCP는 동기 도구를 별도 스레드에서 실행하므로 check_same_thread=False 가 필요하다.
# ---------------------------------------------------------------------------
db = sqlite3.connect(":memory:", check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript(DB_SEED_SQL)


@mcp.tool
def query_database(sql: str) -> list[dict[str, Any]]:
    """쇼핑몰 SQLite 데이터베이스에 읽기 전용 SQL(SELECT)을 실행하고 결과 행을 돌려준다.
    테이블 구조는 리소스 db://schema 에서 확인할 수 있다."""
    if not sql.strip().lower().startswith("select"):
        raise ToolError("SELECT 문만 허용됩니다.")   # ToolError 는 클라이언트에 MCP 오류 응답(isError=True)으로 전달된다

    rows = db.execute(sql).fetchmany(MAX_QUERY_ROWS)
    return [dict(row) for row in rows]


@mcp.resource("db://schema", mime_type="text/plain")
def database_schema() -> str:
    """데이터베이스 테이블 구조 (모델이 SQL을 작성하기 전에 읽는 참고 자료)"""
    rows = db.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(row["sql"] for row in rows)


@mcp.resource("product://{product_id}", mime_type="application/json")
def product_detail(product_id: int) -> dict[str, Any]:
    """상품 ID로 상품 상세 정보를 읽는다. (URI 템플릿 리소스: product://1, product://2 …)"""
    row = db.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    if row is None:
        raise ResourceError(f"상품 {product_id} 을(를) 찾을 수 없습니다.")
    return dict(row)


# ---------------------------------------------------------------------------
# 4. [서버 측] 인터넷 연동 (웹 검색 · HTTP 요청)
#    search_web 은 API 키 없이 실행되도록 검색 결과(SIMULATED_WEB)를 시뮬레이션한다.
#    (실제로는 Tavily/Serper 같은 검색 API를 httpx 로 호출하면 된다. fetch_url 이 그 형태를 보여준다.)
# ---------------------------------------------------------------------------
@mcp.tool
def search_web(query: str) -> list[dict[str, str]]:
    """웹에서 검색어와 관련된 최신 글(제목·URL·요약)을 찾는다. 상품 평판이나 외부 정보가 필요할 때 사용한다."""
    for keyword, results in SIMULATED_WEB.items():
        if keyword in query:
            return results

    return [
        {
            "title": "검색 결과 없음",
            "url": "",
            "snippet": f"'{query}'에 대한 결과를 찾지 못했습니다.",
        },
    ]


@mcp.tool
def fetch_url(url: str) -> str:
    """URL 의 내용을 HTTP GET 으로 가져와 앞부분(최대 500자)을 돌려준다. 실제 네트워크 요청을 수행한다."""
    try:
        response = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        return f"[{response.status_code}] {response.text[:FETCH_PREVIEW_CHARS]}"
    except httpx.HTTPError as e:
        return f"요청 실패: {type(e).__name__}: {e}"   # 오류도 문자열로 돌려주면 모델이 상황을 설명할 수 있다


# ---------------------------------------------------------------------------
# 5. [서버 측] 로컬 파일 연동 (리소스) + 프롬프트 템플릿
# ---------------------------------------------------------------------------
@mcp.resource("docs://readme", mime_type="text/markdown")
def project_readme() -> str:
    """이 프로젝트의 README 앞부분 (파일 시스템 연동 예시)"""
    readme = Path(__file__).with_name("README.md")
    if not readme.exists():
        return "(README 없음)"
    return readme.read_text(encoding="utf-8")[:README_PREVIEW_CHARS]


@mcp.prompt
def sales_briefing(month: str) -> str:
    """특정 월의 판매 현황 브리핑을 작성하도록 지시하는 프롬프트 템플릿"""
    return (
        f"{month}의 판매 데이터를 query_database 도구로 조회해 다음을 보고하세요.\n"
        "1) 가장 많이 팔린 상품 3개와 판매량  2) 카테고리별 매출  3) 눈에 띄는 특이사항\n"
        "SQL 작성 전에 리소스 db://schema 를 먼저 확인하세요."
    )


# ---------------------------------------------------------------------------
# 6. [클라이언트 측] Fake 모델 정의
#    LangChain 에이전트(클라이언트 B)가 사용하는 모델이다.
# ---------------------------------------------------------------------------
class FakeToolCallingModel(GenericFakeChatModel):
    """도구 호출(tool_calls) 응답을 흉내내는 학습용 Fake 모델 (05와 동일)"""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self


def make_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    tool_call = {"name": name, "args": args, "id": call_id, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[tool_call])


# 질문 1개에 대해 "DB 조회 → 웹 검색 → 최종 답변" 순서로 응답한다.
# 도구 이름과 인자는 MCP 서버의 tools/list 에 나오는 스키마와 일치해야 한다.
llm = FakeToolCallingModel(messages=iter([
    make_tool_call(
        "query_database",
        {
            "sql": "SELECT p.name, SUM(o.quantity) AS sold "
                   "FROM orders o JOIN products p ON p.id = o.product_id "
                   "WHERE o.ordered_at BETWEEN '2026-10-01' AND '2026-10-31' "
                   "GROUP BY p.name ORDER BY sold DESC LIMIT 1",
        },
        "call_1",
    ),
    make_tool_call("search_web", {"query": "무선 이어폰 Air 리뷰"}, "call_2"),
    AIMessage(content=(
        "2026년 10월에 가장 많이 팔린 상품은 '무선 이어폰 Air'(42개)입니다. "
        "온라인 반응을 보면 가격 대비 만족도가 높고 착용감 부문 1위라는 평가가 있으며, "
        "노이즈 캔슬링과 통화 품질은 중간 수준이라는 의견이 있습니다."
    )),
]))


# ---------------------------------------------------------------------------
# 7. [클라이언트 측 A] FastMCP Client 로 서버 기능 탐색
#    Client(mcp) 처럼 서버 객체를 직접 넘기면 프로세스를 띄우지 않고 메모리 안에서 MCP 프로토콜로 통신한다.
#    (테스트에 유용. 원격 서버라면 Client("http://host:8000/mcp") 처럼 URL을 넘긴다.)
# ---------------------------------------------------------------------------
async def explore_server() -> None:
    print("=" * 25 + " A. FastMCP Client 로 서버 기능 탐색 (in-memory) " + "=" * 25)

    async with Client(mcp) as client:
        print("\n[tools/list]")
        for tool in await client.list_tools():
            arg_names = ", ".join(tool.inputSchema.get("properties", {}))
            summary = tool.description.splitlines()[0]
            print(f"  - {tool.name}({arg_names}): {summary}")

        print("\n[resources/list + resources/templates/list]")
        for resource in await client.list_resources():
            print(f"  - {resource.uri}  ({resource.mimeType})")
        for template in await client.list_resource_templates():
            print(f"  - {template.uriTemplate}  (템플릿)")

        print("\n[prompts/list]")
        for prompt in await client.list_prompts():
            arg_names = ", ".join(a.name for a in prompt.arguments or [])
            print(f"  - {prompt.name}({arg_names})")

        print("\n[resources/read db://schema]")
        contents = await client.read_resource("db://schema")
        print("  " + contents[0].text.replace("\n", "\n  "))

        print("\n[resources/read product://1]")
        contents = await client.read_resource("product://1")
        print(f"  {json.loads(contents[0].text)}")

        print("\n[tools/call query_database]")
        result = await client.call_tool("query_database", {"sql": DEMO_SQL})
        print(f"  structured: {result.data}")

        print(
            "\n[tools/call query_database — 금지된 문장]  "
            "(서버 로그에 'Error calling tool' 한 줄이 함께 찍히는 것이 정상)"
        )
        result = await client.call_tool(
            "query_database",
            {"sql": FORBIDDEN_SQL},
            raise_on_error=False,
        )
        print(f"  is_error={result.is_error}: {result.content[0].text}")

        print("\n[prompts/get sales_briefing]")
        prompt_result = await client.get_prompt("sales_briefing", {"month": BRIEFING_MONTH})
        print("  " + prompt_result.messages[0].content.text.replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# 8. [클라이언트 측 B] LangChain 에이전트가 MCP 서버의 도구를 사용
# ---------------------------------------------------------------------------
async def run_agent_with_mcp() -> None:
    print("\n\n" + "=" * 25 + " B. LangChain 에이전트 ↔ MCP 서버 (stdio) " + "=" * 25)

    # C1. MCP 서버 연결: 이 파일을 --server 모드의 자식 프로세스로 띄우고 stdio 로 JSON-RPC 통신한다.
    #     여러 서버를 함께 등록할 수 있으며, HTTP 서버는 {"transport": "streamable_http", "url": "..."} 로 적는다.
    mcp_client = MultiServerMCPClient({
        "shop": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(Path(__file__).resolve()), "--server"],
            "env": {"FASTMCP_LOG_LEVEL": "WARNING"},
        },
    })

    # 세션을 열어 두면 서버 프로세스 하나를 계속 재사용한다. (session 없이 get_tools()/도구 호출을 하면 매번 새로 띄운다)
    async with mcp_client.session("shop") as session:
        tools = await load_mcp_tools(session)   # tools/list 결과를 LangChain BaseTool 로 변환
        print(f"\n--- MCP 서버에서 가져온 도구 {len(tools)}개: {[t.name for t in tools]} ---")

        # C2. 에이전트 생성: 도구가 어디서 왔든(파이썬 함수든 MCP 서버든) create_agent 입장에서는 동일하다.
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        print(f"\n[사용자] {USER_QUESTION}")
        response = await agent.ainvoke({"messages": [HumanMessage(content=USER_QUESTION)]})

    # C3. 메시지 흐름: AIMessage(tool_calls) → ToolMessage(MCP 서버 응답) 가 반복된 뒤 최종 AIMessage 가 온다.
    print_message_flow(response["messages"])

    print("\n--- 최종 답변 ---")
    print(response["messages"][-1].content)


# ---------------------------------------------------------------------------
# 9. 실행
#    인자 없이 실행하면 데모(클라이언트 A → B), --server / --http 를 주면 MCP 서버만 실행한다.
# ---------------------------------------------------------------------------
def print_message_flow(messages: list) -> None:
    print("\n--- 메시지 흐름 ---")
    for message in messages:
        role = type(message).__name__

        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                args_text = json.dumps(call["args"], ensure_ascii=False)[:80]
                print(f"[{role}] tools/call 요청 → {call['name']}({args_text}…)")

        elif role == "ToolMessage":   # MCP 응답은 콘텐츠 블록 리스트로 오므로 텍스트만 꺼내 보여준다
            if isinstance(message.content, list):
                text = "".join(b["text"] for b in message.content)
            else:
                text = message.content
            ellipsis = "…" if len(text) > 150 else ""
            print(f"[{role}] MCP 서버 응답 ← {text[:150]}{ellipsis}")

        else:
            print(f"[{role}] {message.content}")


async def main() -> None:
    await explore_server()
    await run_agent_with_mcp()


if __name__ == "__main__":
    if "--server" in sys.argv:
        # stdio 서버: 표준 출력은 프로토콜 채널이므로 print() 를 쓰면 안 된다. (로그는 stderr 로 나간다)
        mcp.run(transport="stdio", show_banner=False)
    elif "--http" in sys.argv:
        # HTTP 서버: 원격 클라이언트는 http://127.0.0.1:8000/mcp 로 접속한다.
        mcp.run(transport="http", host=HTTP_HOST, port=HTTP_PORT)
    else:
        asyncio.run(main())
