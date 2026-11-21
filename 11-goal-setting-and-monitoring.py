"""
11. Goal Setting and Monitoring 패턴

에이전트에게 "무엇을 해야 하는지"를 막연한 지시가 아니라 측정 가능한 목표(goals)와 제약(constraints)으로
명시해 주고, 작업이 진행되는 동안 목표별 달성 여부를 계속 점검(monitoring)해
"계속할지 · 수정할지 · 끝낼지"를 결정하는 패턴.

    목표 설정 (Goal Setting) : 구체적이고(Specific) 측정 가능한(Measurable) 성공 기준을 목록으로 정의한다.
                               반복 횟수 같은 예산(제약)도 함께 정한다.
    모니터링 (Monitoring)    : 매 반복마다 결과물을 목표 목록과 대조해 목표별 달성/미달성을 기록하고 진척도를 계산한다.
    판정 (Decision)          : 모두 달성 → 종료 / 미달성 + 예산 남음 → 피드백을 들고 재시도 / 예산 소진 → 미달성 목표 보고

이 예제는 "주어진 요구사항과 목표 목록을 만족하는 파이썬 코드를 작성하는 코딩 에이전트"다.

흐름:
    M1. 요구사항 + 목표 목록 + 최대 반복 횟수 설정
      → [목표 달성 루프] G1. 코드 생성 (LLM: 요구사항 + 목표 + 이전 코드 + 피드백 → 코드)
                       → G2. 목표 점검 (LLM: 목표별 달성/미달성 + 사유 → 진척도 n/N, 후퇴 감지)
                       → G3. 판정
                             ├─ 미달성 & 반복 남음      → G1로 루프백 (미달성 사유를 피드백으로 전달)
                             └─ 모두 달성 / 최대 반복 도달 → 루프 종료
      → M2. 최종 코드 저장 + 실행 확인

왜 쓰는가:
    - 목표가 명시적 체크리스트로 존재하면 "지금 어디까지 왔는지"(진척도)를 객관적으로 볼 수 있고,
      이전에 달성했던 목표가 다시 깨졌는지(후퇴, regression)도 즉시 드러난다.
    - 종료 조건이 명확하다. "모두 달성" 또는 "예산 소진" 두 경우뿐이므로 에이전트가 무한히 돌지 않는다.
    - 피드백이 "미달성 목표 + 사유"로 구조화되어, 다음 시도가 정확히 무엇을 고쳐야 하는지 안다.

04. Reflection / 09. Learning and Adaptation 과의 차이:
    - Reflection은 검토자의 자유로운 비평으로 품질을 높이고, 종료는 "완벽하다"는 한 마디 신호에 의존한다.
      여기서는 미리 정한 목표 하나하나를 점검하므로 "무엇이 되었고 무엇이 안 되었는지"가 수치로 남는다.
    - Learning(09)은 코드를 실행한 객관적 점수로 더 나은 것만 남기는 진화 방식이고,
      여기서는 목표 달성 여부(체크리스트)로 진행을 관리하며 결과물 하나를 완성해 간다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
생성 모델은 "입력 검증·예시 없음 → 검증·예시 추가(그러나 버그) → 버그 수정" 순서로 세 번 코드를 내고,
점검 모델은 그에 맞춰 목표별 달성/미달성을 돌려준다. (진척도 2/5 → 3/5(목표 2 후퇴) → 5/5)
마지막에 저장한 코드는 실제로 실행해 결과를 확인한다.
(LLM이 만든 코드를 실행한다. 실제 서비스에서는 반드시 샌드박스(컨테이너 등)에서 실행해야 한다.)
"""
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 1. 설정 · 상수 (목표 설정, Goal Setting)
#    요구사항(use case) + 측정 가능한 목표 목록 + 예산(최대 반복 횟수)을 명시한다.
#    목표는 점검자가 "달성/미달성"으로 판정할 수 있을 만큼 구체적이어야 한다.
# ---------------------------------------------------------------------------
# 요구사항과 목표
USE_CASE = "주어진 양의 정수의 Binary Gap(이진 표현에서 1로 둘러싸인 연속된 0의 최대 길이)을 구하는 코드를 작성한다."

GOALS = [
    "이해하기 쉬운 단순한 코드",
    "기능적으로 정확함",
    "경계 조건을 빠짐없이 처리 (간격이 없는 수, 끝이 0으로 끝나는 수 등)",
    "양의 정수만 입력으로 받음 (그 외에는 예외 발생)",
    "몇 가지 예시로 결과를 출력",
]

# 예산과 결과물
MAX_ITERATIONS = 5                                                   # 예산(제약): 이 횟수 안에 끝내야 한다
OUTPUT_FILE = Path(__file__).parent / "output" / "binary_gap.py"     # 최종 결과물 저장 위치
RUN_TIMEOUT_SECONDS = 10                                             # 저장한 코드 실행 확인 시 타임아웃

# LLM 역할 지시문
GENERATOR_ROLE = (
    "당신은 파이썬 코딩 에이전트입니다. 주어진 요구사항을 구현하는 파이썬 코드를 작성하세요. "
    "아래 목표 목록을 모두 만족해야 합니다. 이전 코드와 피드백이 주어지면 그것을 반영해 수정하세요. "
    "설명 없이 ```python 코드 블록 하나만 출력하세요."
)

MONITOR_ROLE = (
    "당신은 코드 검토자입니다. 주어진 코드를 목표 목록과 하나씩 대조해 판정하세요. "
    "목표마다 한 줄씩, 반드시 '번호 | 달성 또는 미달성 | 사유' 형식으로만 답하세요. "
    "미달성이면 무엇을 고쳐야 하는지 사유에 구체적으로 쓰세요."
)

# LLM 응답에서 ```python ... ``` 펜스 안의 코드를 꺼내는 정규식
CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# 2. 목표 점검 결과의 구조
#    점검 모델의 응답을 "목표별 상태"로 파싱해 두면 진척도 계산, 후퇴 감지, 피드백 생성에 모두 쓸 수 있다.
# ---------------------------------------------------------------------------
class GoalStatus(TypedDict):
    index: int      # 목표 번호 (1부터)
    goal: str       # 목표 문장
    met: bool       # 달성 여부
    reason: str     # 판정 사유 (미달성이면 수정 힌트가 된다)


# ---------------------------------------------------------------------------
# 3. Fake 모델 정의
#    생성 모델(코드 작성)과 점검 모델(목표 점검)을 분리해, 루프가 호출하는 순서대로 응답을 배치한다.
# ---------------------------------------------------------------------------
generator_llm = FakeListChatModel(responses=[
    # 반복 1: 로직은 맞지만 입력 검증도, 예시 출력도 없다 → 목표 3·4·5 미달성
    """```python
def binary_gap(n):
    binary = bin(n)[2:]
    max_gap = 0
    current = 0
    for bit in binary:
        if bit == "1":
            max_gap = max(max_gap, current)
            current = 0
        else:
            current += 1
    return max_gap
```""",

    # 반복 2: 검증과 예시를 추가했지만, 끝의 0까지 간격으로 세는 버그가 생겼다 → 목표 2 후퇴
    """```python
def binary_gap(n: int) -> int:
    \"\"\"양의 정수 n의 이진 표현에서 1로 둘러싸인 연속된 0의 최대 길이를 돌려준다.\"\"\"
    if not isinstance(n, int) or n <= 0:
        raise ValueError("양의 정수만 입력할 수 있습니다.")
    return max(len(gap) for gap in bin(n)[2:].split("1"))


if __name__ == "__main__":
    for value in [9, 529, 20, 15]:
        print(f"{value} ({bin(value)[2:]}) → binary gap = {binary_gap(value)}")
```""",

    # 반복 3: 피드백(끝의 0 처리)을 반영해 수정 → 모든 목표 달성
    """```python
def binary_gap(n: int) -> int:
    \"\"\"양의 정수 n의 이진 표현에서 1로 둘러싸인 연속된 0의 최대 길이를 돌려준다.\"\"\"
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("양의 정수만 입력할 수 있습니다.")
    binary = bin(n)[2:].rstrip("0")          # 끝의 0은 1로 닫히지 않으므로 간격이 아니다
    return max((len(gap) for gap in binary.split("1")), default=0)


if __name__ == "__main__":
    for value in [9, 529, 20, 15, 32, 1041, 1]:
        print(f"{value:>5} ({bin(value)[2:]:>11}) → binary gap = {binary_gap(value)}")
```""",
])

monitor_llm = FakeListChatModel(responses=[
    # 반복 1 점검: 2/5
    """1 | 달성 | 단순한 반복문 하나로 구성되어 읽기 쉽다.
2 | 달성 | 1을 만날 때만 최대값을 갱신하므로 9, 529, 20, 15 모두 정답을 돌려준다.
3 | 미달성 | 0이나 음수, 정수가 아닌 값이 들어오면 오류 없이 잘못된 값을 돌려준다.
4 | 미달성 | 입력이 양의 정수인지 검사하는 코드가 없다.
5 | 미달성 | 예시를 실행해 결과를 출력하는 부분이 없다.""",

    # 반복 2 점검: 3/5 (목표 2가 달성 → 미달성으로 후퇴)
    """1 | 달성 | split 한 줄로 간결하게 표현되어 이해하기 쉽다.
2 | 미달성 | 끝의 0도 간격으로 센다. 20(10100)은 1이어야 하는데 2, 32(100000)는 0이어야 하는데 5를 돌려준다.
3 | 미달성 | "끝이 0으로 끝나는 수"와 "간격이 없는 수(예: 32)" 경계 조건이 틀린다.
4 | 달성 | isinstance 와 n <= 0 검사로 양의 정수가 아니면 ValueError 를 낸다.
5 | 달성 | __main__ 블록에서 4개의 예시를 출력한다.""",

    # 반복 3 점검: 5/5
    """1 | 달성 | rstrip 으로 끝의 0을 제거하는 의도가 주석으로 드러나 있고 전체가 짧다.
2 | 달성 | 9→2, 529→4, 20→1, 15→0, 32→0, 1041→5, 1→0 모두 정답이다.
3 | 달성 | 간격이 없는 수(15, 32, 1)와 끝이 0으로 끝나는 수(20, 32)를 올바르게 처리한다.
4 | 달성 | bool 까지 걸러내며, 양의 정수가 아니면 ValueError 를 낸다.
5 | 달성 | 7개의 예시를 이진 표현과 함께 출력한다.""",
])


# ---------------------------------------------------------------------------
# 4. 헬퍼 함수
# ---------------------------------------------------------------------------
def format_goals(goals: list[str]) -> str:
    return "\n".join(f"{i}. {g}" for i, g in enumerate(goals, start=1))


def parse_goal_report(report: str, goals: list[str]) -> list[GoalStatus]:
    """'번호 | 달성/미달성 | 사유' 형식의 점검 응답을 GoalStatus 목록으로 변환한다."""
    statuses: dict[int, GoalStatus] = {}
    for line in report.strip().splitlines():
        parts = [p.strip() for p in line.split("|", maxsplit=2)]
        if len(parts) == 3 and parts[0].isdigit():
            index = int(parts[0])
            statuses[index] = {
                "index": index,
                "goal": goals[index - 1],
                "met": parts[1] == "달성",
                "reason": parts[2],
            }

    # 점검 응답에서 빠진 목표는 보수적으로 "미달성"으로 간주한다.
    return [
        statuses.get(i, {"index": i, "goal": g, "met": False, "reason": "점검 결과 없음"})
        for i, g in enumerate(goals, start=1)
    ]


def clean_code_block(text: str) -> str:
    """LLM 응답에서 ```python ... ``` 펜스를 벗겨 순수 코드만 돌려준다."""
    match = CODE_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 5. 루프의 각 단계
# ---------------------------------------------------------------------------
def generate_code(
    use_case: str,
    goals: list[str],
    previous_code: str,
    feedback: str,
) -> str:
    """[G1 코드 생성] 요구사항 + 목표 (+ 이전 코드 + 피드백) 를 LLM에 주고 코드를 받는다."""
    request = f"[요구사항]\n{use_case}\n\n[목표]\n{format_goals(goals)}"
    if previous_code:
        request += f"\n\n[이전 코드]\n```python\n{previous_code}\n```"
    if feedback:
        request += f"\n\n[이전 코드에 대한 피드백 — 미달성 목표]\n{feedback}"

    response = generator_llm.invoke([
        SystemMessage(content=GENERATOR_ROLE),
        HumanMessage(content=request),
    ])
    return clean_code_block(response.content)


def check_goals(code: str, goals: list[str]) -> list[GoalStatus]:
    """[G2 목표 점검] 코드를 목표 목록과 대조해 목표별 달성 여부를 받는다."""
    request = f"[목표]\n{format_goals(goals)}\n\n[코드]\n```python\n{code}\n```"

    response = monitor_llm.invoke([
        SystemMessage(content=MONITOR_ROLE),
        HumanMessage(content=request),
    ])
    return parse_goal_report(response.content, goals)


def report_progress(statuses: list[GoalStatus], previous: list[GoalStatus] | None) -> None:
    """[G2 모니터링] 목표별 상태, 진척도, 그리고 이전 반복 대비 후퇴한 목표를 출력한다."""
    for s in statuses:
        mark = "✓" if s["met"] else "✗"
        print(f"    {mark} 목표 {s['index']}. {s['goal']}")
        print(f"        └ {s['reason']}")

    met_count = sum(s["met"] for s in statuses)
    bar = "■" * met_count + "□" * (len(statuses) - met_count)
    print(f"--- 진척도: {met_count}/{len(statuses)} 목표 달성 [{bar}] ---")

    if previous:
        regressed = [s for s, p in zip(statuses, previous) if p["met"] and not s["met"]]
        for s in regressed:
            print(f"    ↓ 후퇴 감지: 목표 {s['index']} (달성 → 미달성)")


def build_feedback(statuses: list[GoalStatus]) -> str:
    """미달성 목표와 사유만 모아 다음 생성에 전달할 피드백을 만든다."""
    lines = [
        f"- 목표 {s['index']} ({s['goal']}): {s['reason']}"
        for s in statuses
        if not s["met"]
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. 목표 달성 루프 (Goal Loop)
#    G1 생성 → G2 점검 → G3 판정 을 반복한다. 종료 조건은 "모두 달성" 또는 "최대 반복 도달" 두 가지뿐이다.
# ---------------------------------------------------------------------------
def run_goal_agent(
    use_case: str,
    goals: list[str],
    max_iterations: int,
) -> tuple[str, list[GoalStatus]]:
    code = ""
    feedback = ""
    statuses: list[GoalStatus] = []

    for iteration in range(1, max_iterations + 1):
        feedback_note = " (이전 피드백 반영)" if feedback else ""
        print(f"\n>>> 반복 {iteration}/{max_iterations}: [G1] 코드 생성 중...{feedback_note}")
        code = generate_code(use_case, goals, code, feedback)
        print("    " + code.replace("\n", "\n    "))

        print("\n[G2] 목표 점검 중...")
        previous = statuses
        statuses = check_goals(code, goals)
        report_progress(statuses, previous or None)

        unmet = [s for s in statuses if not s["met"]]
        remaining = max_iterations - iteration
        if not unmet:
            print(f"[G3] 판정: 모든 목표 달성 → 루프 종료 (반복 {iteration}회 사용, {remaining}회 남음)")
            break
        if remaining == 0:
            print(f"[G3] 판정: 최대 반복 도달 → 미달성 목표 {len(unmet)}개를 남기고 종료")
            break

        print(f"[G3] 판정: 미달성 {len(unmet)}개, 남은 반복 {remaining}회 → 피드백을 반영해 재시도")
        feedback = build_feedback(statuses)

    return code, statuses


# ---------------------------------------------------------------------------
# 7. 최종 결과물 저장
# ---------------------------------------------------------------------------
def save_code(
    code: str,
    use_case: str,
    statuses: list[GoalStatus],
    path: Path,
) -> None:
    """최종 코드 앞에 요구사항과 목표 달성 현황을 주석으로 붙여 저장한다."""
    header = [f"# 요구사항: {use_case}", "# 목표 달성 현황:"]
    header += [f"#   [{'v' if s['met'] else ' '}] {s['goal']}" for s in statuses]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header) + "\n\n" + code + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 8. 실행
#    목표 출력 → 목표 달성 루프 → 최종 코드 저장 → 저장한 코드 실행 확인
# ---------------------------------------------------------------------------
def main() -> None:
    print("## 목표 설정·모니터링 코딩 에이전트 실행 ##")
    print(f"\n[M1] 요구사항: {USE_CASE}")
    print("[M1] 목표:")
    print("    " + format_goals(GOALS).replace("\n", "\n    "))
    print(f"[M1] 예산: 최대 {MAX_ITERATIONS}회 반복")

    code, statuses = run_goal_agent(USE_CASE, GOALS, MAX_ITERATIONS)

    print(f"\n[M2] 최종 코드 저장: {OUTPUT_FILE.relative_to(Path(__file__).parent)}")
    save_code(code, USE_CASE, statuses, OUTPUT_FILE)

    print("[M2] 저장한 코드 실행 확인:")
    result = subprocess.run(
        [sys.executable, str(OUTPUT_FILE)],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    output = result.stdout or result.stderr
    print("    " + output.rstrip().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
