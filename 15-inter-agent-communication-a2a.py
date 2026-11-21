"""
15. Inter-Agent Communication (A2A) 패턴

서로 다른 팀·프레임워크·서버에서 만든 에이전트들이 "공통 규약"으로 대화하며 일을 나눠 맡는 패턴.
A2A(Agent2Agent)는 Google 이 제안한 에이전트 간 통신용 개방형 프로토콜로, HTTP + JSON-RPC 2.0 위에서 동작한다.

    에이전트 카드 (Agent Card) : 원격 에이전트가 "나는 누구이고 무엇을 할 수 있는지"를 공개하는 명함(JSON).
                                 /.well-known/agent-card.json 에 두면 클라이언트가 찾아 읽는다. (Discovery)
    스킬 (Skill)               : 카드에 적힌 능력 목록. 클라이언트는 이것을 보고 어느 에이전트에게 맡길지 고른다.
    메시지 (Message)           : 에이전트끼리 주고받는 한 번의 발화. role(user/agent) + parts(text 등) 로 이루어진다.
    태스크 (Task)              : 요청 하나의 작업 단위. id 와 상태(state)를 가지고, 결과는 artifacts 에 담긴다.
                                 상태: submitted → working → completed / failed / input-required(추가 정보 필요)
    컨텍스트 (contextId)       : 여러 메시지·태스크를 하나의 대화로 묶는 id. 이어서 말할 때 같은 값을 보낸다.

이 예제는 "개인 비서 에이전트(클라이언트)가 원격 캘린더 에이전트(서버)에게 일정 확인을 맡기는" 구조다.
    - 서버: Calendar Agent — 에이전트 카드를 공개하고, JSON-RPC 요청을 받아 일정을 확인한다.
    - 클라이언트: Personal Assistant — 사용자 요청을 받아 카드의 스킬을 보고 원격 에이전트에게 위임한다.

흐름:
    M1. 탐색 (Discovery) : 클라이언트가 GET /.well-known/agent-card.json 으로 원격 에이전트 카드를 읽는다.
    M2. 사용자 요청 입력
      → [A2A 통신] A1. 스킬 매칭 : 요청 문장과 카드의 skills(tags)를 비교해 맡길 수 있는지 판단
                     ├─ 맞는 스킬 없음 → 원격 호출 없이 비서가 직접 "처리할 수 없다"고 답변
                     └─ 맞는 스킬 있음 → A2. message/send : JSON-RPC 로 메시지 전송
                                          → (서버) A3. 태스크 생성 · 캘린더 에이전트 실행 → Task 반환
                                          → A4. 태스크 상태 확인
                                               ├─ input-required → 사용자에게 추가 정보를 받아 같은 taskId 로 재전송 (A2)
                                               ├─ completed      → artifacts 에서 결과 추출
                                               └─ failed         → 오류 안내
      → M3. 결과 출력

왜 쓰는가:
    - 에이전트를 만든 프레임워크(ADK, LangGraph, CrewAI ...)가 달라도 같은 규약으로 협업할 수 있다.
    - 클라이언트는 원격 에이전트의 내부 구현을 몰라도 된다. 카드(능력)와 프로토콜(주고받는 형식)만 알면 된다.
    - 태스크에 상태가 있어 "추가 정보가 필요하다(input-required)" 같은 여러 턴짜리 협업을 표현할 수 있다.

07. Multi-Agent Collaboration / 10. MCP 와의 차이:
    - 07 은 한 프로그램(한 그래프) 안의 에이전트들이 공유 상태로 협업한다. A2A 는 서로 다른 서버의 에이전트가
      네트워크 너머로 메시지를 주고받는다.
    - MCP(10)는 에이전트 ↔ 도구·데이터 연결 규약이고, A2A 는 에이전트 ↔ 에이전트 협업 규약이다.
      원격 에이전트는 도구처럼 한 번 호출되고 끝나는 것이 아니라, 스스로 판단하고 되묻기도 한다.

원래 Google ADK 스케치와의 대응:
    - 스케치는 ADK LlmAgent(Google Calendar 도구) 를 a2a-sdk 의 A2AStarletteApplication 으로 감싸 서버로 띄웠다.
    - 여기서는 프로토콜 자체를 보기 위해 A2A 핵심 규약(에이전트 카드, message/send, tasks/get, 태스크 상태)을
      Starlette(서버) + httpx(클라이언트) 로 직접 구현했다. AgentSkill · AgentCard 의 필드 구성은 스케치와 같다.
    - Google Calendar API 대신 가짜 일정 데이터(CALENDAR_EVENTS)를 쓰고, LLM 대신 규칙으로 요청을 해석한다.
    - 실제 서비스에서는 공식 SDK(a2a-sdk)를 쓰면 이 서버·클라이언트 코드를 SDK 가 대신 처리해 준다.
      (SDK 는 1.x 에서 API 가 크게 바뀌었으므로 버전에 맞는 문서를 확인할 것)

학습용이므로 실제 API 대신 가짜 일정 데이터와 규칙 기반 에이전트를 사용한다.
기본 실행은 서버를 프로세스 안에서 띄우고(httpx.ASGITransport) 실제 HTTP 와 같은 요청·응답을 주고받는다.
    1) 내일 오전 10시부터 11시까지 비어 있어?  → completed (팀 주간 회의와 겹침)
    2) 내일 오후 4시부터 5시까지 시간 돼?      → completed (비어 있음)
    3) 모레 일정 알려줘                        → completed (일정 목록 스킬)
    4) 내일 시간 비어 있어?                    → input-required → "오후 1시부터 2시까지" → completed
    5) 오늘 날씨 어때?                         → 맞는 스킬 없음 → 원격 호출 없이 비서가 직접 답변
`python 15-inter-agent-communication-a2a.py --demo`   : 위 예시를 정해진 응답으로 자동 실행
`python 15-inter-agent-communication-a2a.py --server` : 캘린더 에이전트를 http://127.0.0.1:10000 에 실제로 띄움
    (다른 터미널에서 curl http://127.0.0.1:10000/.well-known/agent-card.json 으로 카드를 볼 수 있다)
"""
import asyncio
import datetime
import json
import re
import sys
import uuid
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 서버 주소 (--server 모드)
HOST = "127.0.0.1"
PORT = 10000
AGENT_URL = f"http://{HOST}:{PORT}/"
AGENT_CARD_PATH = "/.well-known/agent-card.json"   # A2A 가 정한 카드 위치 (Discovery)

# 에이전트 카드: 원격 에이전트의 명함. (스케치의 AgentSkill · AgentCard 와 같은 필드)
#   학습용으로 skills 의 tags 를 "이 스킬이 반응하는 키워드"로 사용한다.
AGENT_CARD = {
    "name": "Calendar Agent",
    "description": "사용자의 캘린더를 확인해 빈 시간과 일정을 알려주는 에이전트",
    "url": AGENT_URL,
    "version": "1.0.0",
    "protocolVersion": "0.3.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "check_availability",
            "name": "Check Availability",
            "description": "특정 시간대에 사용자가 비어 있는지 캘린더로 확인한다.",
            "tags": ["비어", "시간 돼", "가능", "빈 시간"],
            "examples": ["내일 오전 10시부터 11시까지 비어 있어?"],
        },
        {
            "id": "list_events",
            "name": "List Events",
            "description": "특정 날짜의 일정 목록을 알려준다.",
            "tags": ["일정", "스케줄"],
            "examples": ["모레 일정 알려줘"],
        },
    ],
}

# 가짜 캘린더 데이터 (Google Calendar API 대신)
TODAY = datetime.date(2026, 11, 23)
CALENDAR_EVENTS = {
    "2026-11-23": [("09:00", "10:00", "스탠드업 미팅")],
    "2026-11-24": [
        ("10:00", "11:00", "팀 주간 회의"),
        ("14:00", "15:30", "고객 미팅"),
    ],
    "2026-11-25": [
        ("11:00", "12:00", "디자인 리뷰"),
        ("16:00", "17:00", "1:1 면담"),
    ],
}

# 요청 해석 규칙
DAY_OFFSETS = {"오늘": 0, "내일": 1, "모레": 2}
TIME_PATTERN = re.compile(r"(오전|오후)?\s*(\d{1,2})시(?:\s*(\d{1,2})분|\s*(반))?")
LIST_KEYWORDS = ("일정", "스케줄")
ASK_TIME_MESSAGE = "몇 시부터 몇 시까지 확인할까요? (예: 오후 1시부터 2시까지)"

# JSON-RPC 2.0 오류 코드 (-32001 은 A2A 가 정의한 TaskNotFound)
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
TASK_NOT_FOUND = -32001

# 데모: (사용자 요청, input-required 일 때 사용자가 할 추가 답변 목록)
DEMO_CASES = [
    ("내일 오전 10시부터 11시까지 비어 있어?", []),
    ("내일 오후 4시부터 5시까지 시간 돼?", []),
    ("모레 일정 알려줘", []),
    ("내일 시간 비어 있어?", ["오후 1시부터 2시까지"]),
    ("오늘 날씨 어때?", []),
]


# ---------------------------------------------------------------------------
# 2. A2A 데이터 구조 (Message · Task)
#    A2A 에서 주고받는 객체는 모두 JSON 이다. 여기서는 dict 를 만드는 함수로 형식을 보여준다.
# ---------------------------------------------------------------------------
def new_id() -> str:
    return uuid.uuid4().hex[:8]


def make_message(
    role: str,
    text: str,
    task_id: str | None = None,
    context_id: str | None = None,
) -> dict:
    """Message: role(user/agent) + parts. 이어지는 대화라면 taskId · contextId 를 함께 보낸다."""
    message = {
        "kind": "message",
        "messageId": new_id(),
        "role": role,
        "parts": [{"kind": "text", "text": text}],
    }
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id
    return message


def make_task(context_id: str) -> dict:
    """Task: 요청 하나의 작업 단위. 처음에는 submitted 상태로 만들어진다."""
    return {
        "kind": "task",
        "id": new_id(),
        "contextId": context_id,
        "status": {"state": "submitted"},
        "artifacts": [],
        "history": [],
    }


def message_text(message: dict) -> str:
    """Message 의 text part 들을 이어 붙인다."""
    texts = [part["text"] for part in message["parts"] if part["kind"] == "text"]
    return " ".join(texts)


# ---------------------------------------------------------------------------
# 3. [서버 측] 캘린더 에이전트 로직
#    실제로는 LLM + Google Calendar 도구가 하는 일이다. 여기서는 규칙으로 요청을 해석한다.
#    A2A 서버 입장에서는 "메시지를 받아 태스크 상태와 결과를 채우는 무언가"이면 무엇이든 된다.
# ---------------------------------------------------------------------------
def parse_day(text: str) -> datetime.date:
    """'오늘/내일/모레' 를 날짜로 바꾼다. 없으면 오늘로 본다."""
    for word, offset in DAY_OFFSETS.items():
        if word in text:
            return TODAY + datetime.timedelta(days=offset)
    return TODAY


def parse_times(text: str) -> list[datetime.time]:
    """'오전 10시', '오후 4시 30분', '1시 반' 같은 표현을 시각 목록으로 바꾼다.

    오전/오후가 없으면 앞 시각의 오전/오후를 따르고, 맨 앞이면 1~7시는 오후로 본다.
    """
    times = []
    meridiem = None

    for marker, hour_text, minute_text, half in TIME_PATTERN.findall(text):
        meridiem = marker or meridiem
        hour = int(hour_text)
        minute = 30 if half else int(minute_text or 0)

        if meridiem == "오후" and hour < 12:
            hour += 12
        elif meridiem is None and hour < 8:
            hour += 12

        times.append(datetime.time(hour, minute))

    return times


def to_time(text: str) -> datetime.time:
    return datetime.time.fromisoformat(text)


def check_availability(day: datetime.date, start: datetime.time, end: datetime.time) -> str:
    """[스킬 check_availability] 시간대가 기존 일정과 겹치는지 확인한다."""
    events = CALENDAR_EVENTS.get(day.isoformat(), [])
    conflicts = [
        title
        for event_start, event_end, title in events
        if start < to_time(event_end) and to_time(event_start) < end
    ]

    period = f"{day.isoformat()} {start:%H:%M}~{end:%H:%M}"
    if conflicts:
        return f"{period} 에는 '{', '.join(conflicts)}' 일정이 있어 비어 있지 않습니다."
    return f"{period} 은(는) 비어 있습니다."


def list_events(day: datetime.date) -> str:
    """[스킬 list_events] 그날의 일정 목록을 만든다."""
    events = CALENDAR_EVENTS.get(day.isoformat(), [])
    if not events:
        return f"{day.isoformat()} 에는 일정이 없습니다."

    lines = [f"{start}~{end} {title}" for start, end, title in events]
    return f"{day.isoformat()} 일정: " + " / ".join(lines)


def run_calendar_agent(task: dict) -> None:
    """태스크의 대화 기록(history)을 읽고 상태와 결과를 채운다. (A2A 의 AgentExecutor 역할)

    input-required 뒤에 온 추가 답변도 같은 태스크의 history 에 쌓이므로, 사용자 발화를 모두 합쳐 해석한다.
    """
    task["status"] = {"state": "working"}

    user_texts = [message_text(m) for m in task["history"] if m["role"] == "user"]
    request = " ".join(user_texts)
    day = parse_day(request)

    # 스킬 1: 일정 목록
    if any(keyword in request for keyword in LIST_KEYWORDS):
        complete_task(task, list_events(day))
        return

    # 스킬 2: 빈 시간 확인 — 시간이 없으면 되묻는다 (input-required)
    times = parse_times(request)
    if not times:
        reply = make_message("agent", ASK_TIME_MESSAGE, task["id"], task["contextId"])
        task["history"].append(reply)
        task["status"] = {"state": "input-required", "message": reply}
        return

    start = times[0]
    if len(times) > 1:
        end = times[1]
    else:
        end = datetime.time(min(start.hour + 1, 23), start.minute)   # 끝 시각이 없으면 1시간으로 본다

    complete_task(task, check_availability(day, start, end))


def complete_task(task: dict, result_text: str) -> None:
    """결과를 artifact 로 담고 태스크를 completed 로 바꾼다."""
    artifact = {
        "artifactId": new_id(),
        "name": "calendar_result",
        "parts": [{"kind": "text", "text": result_text}],
    }
    task["artifacts"].append(artifact)
    task["status"] = {"state": "completed"}


# ---------------------------------------------------------------------------
# 4. [서버 측] A2A 서버 (Starlette)
#    GET  /.well-known/agent-card.json → 에이전트 카드
#    POST /                            → JSON-RPC 2.0 (message/send, tasks/get)
# ---------------------------------------------------------------------------
_task_store: dict[str, dict] = {}   # taskId → Task (실제로는 DB 에 저장한다)


def public_task(task: dict) -> dict:
    """응답에는 history 를 빼고 보낸다. (전체 기록은 tasks/get 으로 조회)"""
    return {key: value for key, value in task.items() if key != "history"}


def jsonrpc_result(request_id: Any, result: dict) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def jsonrpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    error = {"code": code, "message": message}
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": error})


async def get_agent_card(_request: Request) -> JSONResponse:
    return JSONResponse(AGENT_CARD)


async def handle_jsonrpc(request: Request) -> JSONResponse:
    body = await request.json()
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    if method == "message/send":
        return handle_message_send(request_id, params)
    if method == "tasks/get":
        return handle_tasks_get(request_id, params)
    return jsonrpc_error(request_id, METHOD_NOT_FOUND, f"지원하지 않는 메서드입니다: {method}")


def handle_message_send(request_id: Any, params: dict) -> JSONResponse:
    """[A3] 메시지를 받아 태스크를 만들거나(새 요청) 이어서 처리한다(taskId 가 있는 추가 답변)."""
    message = params.get("message")
    if not message or not message.get("parts"):
        return jsonrpc_error(request_id, INVALID_PARAMS, "params.message.parts 가 필요합니다.")

    task_id = message.get("taskId")
    if task_id:
        task = _task_store.get(task_id)
        if task is None:
            return jsonrpc_error(request_id, TASK_NOT_FOUND, f"태스크를 찾을 수 없습니다: {task_id}")
    else:
        context_id = message.get("contextId") or new_id()
        task = make_task(context_id)
        _task_store[task["id"]] = task

    task["history"].append(message)
    run_calendar_agent(task)
    return jsonrpc_result(request_id, public_task(task))


def handle_tasks_get(request_id: Any, params: dict) -> JSONResponse:
    task = _task_store.get(params.get("id", ""))
    if task is None:
        return jsonrpc_error(request_id, TASK_NOT_FOUND, f"태스크를 찾을 수 없습니다: {params.get('id')}")
    return jsonrpc_result(request_id, task)


calendar_agent_app = Starlette(routes=[
    Route(AGENT_CARD_PATH, get_agent_card, methods=["GET"]),
    Route("/", handle_jsonrpc, methods=["POST"]),
])


# ---------------------------------------------------------------------------
# 5. [클라이언트 측] A2A 클라이언트
#    원격 에이전트의 주소만 알면 카드를 읽고, JSON-RPC 로 메시지를 보낼 수 있다.
# ---------------------------------------------------------------------------
class A2AClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.card: dict = {}
        self._request_seq = 0

    async def discover(self) -> dict:
        """[M1] GET /.well-known/agent-card.json 으로 에이전트 카드를 읽는다."""
        print(f"→ GET {AGENT_CARD_PATH}")
        response = await self.http.get(AGENT_CARD_PATH)
        response.raise_for_status()
        self.card = response.json()
        return self.card

    def find_skill(self, text: str) -> dict | None:
        """[A1] 요청 문장에 카드의 스킬 tags 가 들어 있으면 그 스킬을 돌려준다."""
        for skill in self.card["skills"]:
            if any(tag in text for tag in skill["tags"]):
                return skill
        return None

    async def send_message(self, message: dict) -> dict:
        """[A2] JSON-RPC message/send 로 메시지를 보내고 Task 를 받는다."""
        return await self._call("message/send", {"message": message})

    async def _call(self, method: str, params: dict) -> dict:
        self._request_seq += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_seq,
            "method": method,
            "params": params,
        }
        print(f"→ POST / {method}")
        print_json(params)

        response = await self.http.post("/", json=payload)
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"A2A 오류 {body['error']['code']}: {body['error']['message']}")

        print(f"← result ({method})")
        print_json(body["result"])
        return body["result"]


# ---------------------------------------------------------------------------
# 6. [클라이언트 측] 개인 비서 에이전트
#    사용자 요청을 받아 원격 에이전트에게 맡기고, 태스크 상태에 따라 다음 행동을 정한다.
# ---------------------------------------------------------------------------
def artifact_text(task: dict) -> str:
    texts = [
        part["text"]
        for artifact in task["artifacts"]
        for part in artifact["parts"]
        if part["kind"] == "text"
    ]
    return " ".join(texts)


async def delegate(client: A2AClient, user_request: str, ask_user) -> str:
    """요청 하나를 처리한다. input-required 가 오면 사용자에게 묻고 같은 태스크로 이어서 보낸다."""
    skill = client.find_skill(user_request)
    if skill is None:
        print(f"[A1] 스킬 매칭: '{client.card['name']}' 카드에 맞는 스킬 없음 → 원격 호출 생략")
        return "죄송합니다. 연결된 에이전트 중 이 요청을 처리할 수 있는 에이전트가 없습니다."

    print(f"[A1] 스킬 매칭: {skill['id']} → '{client.card['name']}' 에게 위임\n")

    print("[A2] message/send")
    task = await client.send_message(make_message("user", user_request))

    while True:
        state = task["status"]["state"]
        print(f"\n[A4] 태스크 상태: {state}")

        if state == "completed":
            return artifact_text(task)
        if state == "failed":
            return "원격 에이전트가 작업에 실패했습니다."
        if state != "input-required":
            return f"처리할 수 없는 태스크 상태입니다: {state}"

        # input-required: 원격 에이전트의 질문을 사용자에게 전달하고, 답을 같은 taskId 로 보낸다.
        question = message_text(task["status"]["message"])
        answer = ask_user(question)

        print("\n[A2] message/send (같은 taskId · contextId 로 이어서 전송)")
        follow_up = make_message("user", answer, task["id"], task["contextId"])
        task = await client.send_message(follow_up)


# ---------------------------------------------------------------------------
# 7. 실행
# ---------------------------------------------------------------------------
def print_json(data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    print("    " + text.replace("\n", "\n    "))


def print_card(card: dict) -> None:
    print(f"← 에이전트 카드: {card['name']} v{card['version']} ({card['url']})")
    print(f"    {card['description']}")
    for skill in card["skills"]:
        print(f"    - 스킬 {skill['id']}: {skill['description']} (tags: {', '.join(skill['tags'])})")


def ask_user_interactive(question: str) -> str:
    print(f"\n    ┌─ 원격 에이전트의 질문: {question}")
    return input("    └─ 답변> ").strip()


def scripted_user(answers: list[str]):
    """--demo 용: 정해진 답변을 차례로 돌려준다."""
    def respond_with(question: str) -> str:
        answer = answers.pop(0)
        print(f"\n    ┌─ 원격 에이전트의 질문: {question}")
        print(f"    └─ (데모 답변) {answer}")
        return answer
    return respond_with


async def handle_request(client: A2AClient, user_request: str, ask_user) -> None:
    print(f"\n[M2] 사용자 요청: {user_request}")
    try:
        answer = await delegate(client, user_request, ask_user)
    except (httpx.HTTPError, RuntimeError) as e:   # 네트워크 · 프로토콜 오류
        answer = f"원격 에이전트와 통신하지 못했습니다: {e}"
    print(f"\n[M3] 비서 답변: {answer}")


async def run_client(demo: bool) -> None:
    print("## A2A 에이전트 간 통신: 개인 비서 → 캘린더 에이전트 ##")
    print(f"(오늘 날짜: {TODAY.isoformat()}, 서버는 프로세스 안에서 실행 — 실제 HTTP 요청·응답과 같은 형식)\n")

    # 서버를 프로세스 안에서 띄운다. 실제 원격 서버라면 transport 없이 base_url 만 바꾸면 된다.
    transport = httpx.ASGITransport(app=calendar_agent_app)
    async with httpx.AsyncClient(transport=transport, base_url=AGENT_URL) as http:
        client = A2AClient(http)

        print("[M1] 탐색 (Discovery)")
        print_card(await client.discover())

        if demo:
            for user_request, answers in DEMO_CASES:
                print("\n" + "=" * 80)
                await handle_request(client, user_request, scripted_user(list(answers)))
            return

        print("\n일정에 대해 요청하세요. (종료: q)")
        print("예시:\n    " + "\n    ".join(request for request, _ in DEMO_CASES))
        while True:
            try:
                user_request = input("\n요청> ").strip()
                if user_request.lower() in {"q", "quit", "exit"}:
                    break
                if user_request:
                    await handle_request(client, user_request, ask_user_interactive)
            except (EOFError, KeyboardInterrupt):     # Ctrl+D / Ctrl+C 도 오류 없이 종료
                print()
                break
        print("에이전트를 종료합니다.")


def main() -> None:
    if "--server" in sys.argv:
        print(f"Calendar Agent 를 {AGENT_URL} 에서 실행합니다. (카드: {AGENT_URL.rstrip('/')}{AGENT_CARD_PATH})")
        uvicorn.run(calendar_agent_app, host=HOST, port=PORT)
        return

    asyncio.run(run_client(demo="--demo" in sys.argv))


if __name__ == "__main__":
    main()
