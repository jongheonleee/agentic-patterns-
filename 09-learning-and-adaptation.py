"""
09. Learning and Adaptation 패턴

에이전트가 한 번 만든 결과에 머무르지 않고, 경험(평가 결과·피드백)을 바탕으로
스스로 행동을 바꿔 가며 성능을 점점 높이는 패턴.

이 예제는 그중 "진화형 코딩 에이전트(Evolutionary Coding Agent)" 방식을 구현한다.
OpenEvolve(AlphaEvolve의 오픈소스 구현)는 LLM을 활용해 코드를 반복적으로 최적화하는데,
핵심 루프는 다음 세 단계의 반복이다.

    변이 (Mutate)   : LLM이 부모 프로그램 + 평가 지표 + 실패 피드백을 보고 개선된 자식 프로그램을 제안
    평가 (Evaluate) : 자식 프로그램을 실제로 실행해 정확도·효율 등 지표(metrics)를 측정  (= 보상 신호)
    선택 (Select)   : 지금까지의 최고 프로그램보다 점수가 높으면 채택, 아니면 기각

흐름:
    초기 프로그램 + 평가기 준비
      → [진화 루프] T1. 변이  → T2. 평가 → T3. 선택 (채택: best 갱신 / 기각: 피드백만 보관)
            ├─ 세대 남음       → T1로 루프백
            └─ 최대 세대 도달   → 종료
      → 최고 프로그램과 지표 출력

왜 쓰는가:
    - 정답 코드를 LLM이 한 번에 쓰게 하는 대신, "실행해 보고 → 점수로 비교하고 → 더 나은 것만 남기는"
      과정을 반복하면 LLM의 실수(버그, 비효율)가 평가기에서 걸러지고, 최고 프로그램의 점수는 결코 나빠지지 않는다.
    - 사람이 설계하는 것은 "무엇이 좋은 프로그램인가"를 정의하는 평가기뿐이다.
      보상(점수) 설계가 진화 방향을 결정하므로, 지표를 어떻게 잡느냐가 가장 중요하다.

04. Reflection / 08. Memory Management 와의 차이:
    - Reflection은 LLM의 "자기 평가"에 의존하지만, 여기서는 코드를 실제로 실행한 "객관적 점수"로 평가한다.
    - Memory는 사실을 저장해 다음에 꺼내 쓰는 것이고, Learning은 그 경험으로 행동(코드) 자체를 바꾼다.

LangGraph 사용 포인트:
    - 세대(generation)마다 mutate → evaluate → select 노드를 거치고, 조건부 엣지로 루프/종료를 결정한다.
    - 상태의 history 키는 operator.add 리듀서로 누적되어 세대별 기록(아카이브) 역할을 한다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
Fake 모델은 세대마다 "메모이제이션 → (버그 있는) 반복문 → 올바른 반복문 → 재귀 고속 배가" 순서로
자식 프로그램을 제안한다. 평가기는 진짜로 코드를 실행하므로 점수와 채택/기각은 실제 계산 결과다.
(LLM이 만든 코드를 exec()로 실행한다. 실제 서비스에서는 반드시 샌드박스(컨테이너 등)에서 실행해야 한다.)
"""
import math
import operator
import re
import sys
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 초기 프로그램 (initial_program)
# 진화의 출발점. 정확하지만 지수 시간이 걸리는 순진한 재귀 구현이다.
INITIAL_PROGRAM = '''\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
'''
INITIAL_DESCRIPTION = "초기 프로그램 (순진한 재귀)"

# 진화 루프 설정
MAX_GENERATIONS = 4   # OpenEvolve의 iterations 에 해당

# 평가기 설정
TEST_CASES = [(0, 0), (1, 1), (2, 1), (5, 5), (10, 55), (20, 6765)]
BENCHMARK_N = 22
EVOLVED_FILENAME = "<evolved>"   # 진화 대상 코드를 컴파일할 때 붙이는 파일 이름 (호출 횟수 측정에 활용)

MUTATOR_ROLE = (
    "당신은 코드 최적화 전문가입니다. 주어진 '현재 최고 프로그램'을 더 정확하고 효율적으로 개선한 "
    "새 버전을 제안하세요. 함수 이름과 시그니처(fib(n))는 유지해야 합니다. "
    "첫 줄에 '변이 설명: ...' 을 쓰고, 그 아래 ```python 코드 블록으로 전체 프로그램을 출력하세요."
)


# ---------------------------------------------------------------------------
# 2. 상태(State) 정의
# ---------------------------------------------------------------------------
class Program(TypedDict):
    code: str
    metrics: dict
    description: str


class EvolutionState(TypedDict):
    generation: int                              # 현재 세대 번호
    best: Program                                # 지금까지의 최고 프로그램
    candidate: Program                           # 이번 세대에 LLM이 제안한 자식 프로그램
    feedback: list[str]                          # 최근 평가에서 나온 실패 피드백 (다음 변이에 전달)
    accepted: bool                               # 이번 세대 채택 여부
    history: Annotated[list[dict], operator.add] # 세대별 기록 (아카이브), 누적
    best_program: Program                        # 종료 시 보고용 (그래프 밖에서 쓰는 결과)


# ---------------------------------------------------------------------------
# 3. Fake 모델 정의
#    변이 단계에서 LLM이 제안하는 자식 프로그램. 실제로는 부모 코드와 지표, 피드백을 읽고 개선안을 쓴다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # 세대 1: 중복 계산 제거 (메모이제이션) → 정확 + 호출 수 급감 → 채택
    """변이 설명: 중복 재귀 호출을 lru_cache 메모이제이션으로 제거했습니다.
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
```""",

    # 세대 2: 반복문으로 전환했지만 반환값이 한 칸 밀린 버그 → 테스트 실패 → 기각
    """변이 설명: 재귀를 없애고 반복문으로 바꿔 함수 호출을 1회로 줄였습니다.
```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return b
```""",

    # 세대 3: 피드백(기대값/실제값)을 반영해 반환값 수정 → 정확 + 호출 1회 → 채택
    """변이 설명: 피드백을 반영해 반환값을 b가 아닌 a로 고쳤습니다. (fib(0)=0 이 되도록)
```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```""",

    # 세대 4: 점근적으로 더 빠른 고속 배가(fast doubling) 재귀. 정확하지만 n=22 기준 호출 수는 더 많음 → 기각
    """변이 설명: O(log n) 고속 배가(fast doubling) 알고리즘으로 교체했습니다.
```python
def fib(n):
    def fib_pair(k):
        if k == 0:
            return (0, 1)
        a, b = fib_pair(k // 2)
        c = a * (2 * b - a)
        d = a * a + b * b
        return (c, d) if k % 2 == 0 else (d, c + d)
    return fib_pair(n)[0]
```""",
])


# ---------------------------------------------------------------------------
# 4. 평가기 (evaluator) · 헬퍼 함수
#    프로그램을 실제로 실행해 지표를 계산한다. OpenEvolve의 evaluation_file 역할.
#    - correctness : 테스트 통과율 (0~1)
#    - calls       : fib(22) 계산에 든 함수 호출 횟수 (적을수록 효율적, 결정적으로 측정됨)
#    - combined    : 두 지표를 합친 종합 점수. 선택(select) 단계는 이 값만 본다.
# ---------------------------------------------------------------------------
def load_program(code: str) -> dict:
    """코드 문자열을 격리된 네임스페이스에서 실행해 정의된 이름들을 돌려준다."""
    namespace: dict = {}
    exec(compile(code, EVOLVED_FILENAME, "exec"), namespace)   # filename을 지정해 호출 횟수 측정에 활용
    return namespace


def count_calls(fn, n: int) -> int:
    """fn(n)을 실행하는 동안 진화 대상 프로그램 안에서 일어난 파이썬 함수 호출 횟수를 센다."""
    calls = 0

    def profiler(frame, event, _):
        nonlocal calls
        if event == "call" and frame.f_code.co_filename == EVOLVED_FILENAME:
            calls += 1

    sys.setprofile(profiler)
    try:
        fn(n)
    finally:
        sys.setprofile(None)
    return calls


def evaluate_program(code: str) -> tuple[dict, list[str]]:
    """프로그램을 평가해 (metrics, feedback) 을 돌려준다. feedback은 다음 변이 때 LLM에 전달된다."""
    feedback: list[str] = []
    try:
        fib = load_program(code)["fib"]
        passed = 0
        for n, expected in TEST_CASES:
            actual = fib(n)
            if actual == expected:
                passed += 1
            else:
                feedback.append(f"fib({n}) 기대값 {expected}, 실제값 {actual}")
        correctness = passed / len(TEST_CASES)

        # 효율은 캐시 등 내부 상태가 비어 있는 "새로 로드한" 프로그램으로 측정한다.
        calls = count_calls(load_program(code)["fib"], BENCHMARK_N)
    except Exception as e:   # 문법 오류, 실행 오류 등
        feedback.append(f"실행 오류: {type(e).__name__}: {e}")
        correctness = 0.0
        calls = math.inf

    if calls == math.inf:
        efficiency = 0.0
    else:
        efficiency = 1 / (1 + math.log10(max(calls, 1)))   # 호출 1회 이하면 1.0

    metrics = {
        "correctness": correctness,
        "calls": calls,
        "combined": round(0.7 * correctness + 0.3 * efficiency, 3),
    }
    return metrics, feedback


def parse_mutation(response: str) -> tuple[str, str]:
    """LLM 응답에서 (변이 설명, 코드) 를 추출한다."""
    description = response.strip().splitlines()[0].replace("변이 설명:", "").strip()
    match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    code = match.group(1) if match else response
    return description, code


# ---------------------------------------------------------------------------
# 5. 노드 정의
# ---------------------------------------------------------------------------
def mutate_node(state: EvolutionState) -> dict:
    """[T1 변이] 최고 프로그램 + 지표 + 이전 피드백을 LLM에 보여주고 개선된 자식 프로그램을 받는다."""
    generation = state["generation"] + 1
    best = state["best"]
    best_metrics = best["metrics"]
    print(f"\n>>> 세대 {generation}: [T1] 변이 중... (부모 combined={best_metrics['combined']})")

    request = (
        f"[현재 최고 프로그램] (combined={best_metrics['combined']}, "
        f"correctness={best_metrics['correctness']:.2f}, calls={best_metrics['calls']})\n"
        f"```python\n{best['code']}```"
    )
    if state.get("feedback"):   # 직전 후보가 실패했다면 그 이유를 함께 알려준다 (OpenEvolve의 artifacts)
        request += "\n\n[직전 제안의 실패 피드백]\n" + "\n".join(f"- {f}" for f in state["feedback"])

    response = llm.invoke([
        SystemMessage(content=MUTATOR_ROLE),
        HumanMessage(content=request),
    ])
    description, code = parse_mutation(response.content)
    print(f"--- 제안: {description} ---")
    print("    " + code.strip().replace("\n", "\n    "))

    candidate = {
        "code": code,
        "metrics": {},
        "description": description,
    }
    return {"generation": generation, "candidate": candidate}


def evaluate_node(state: EvolutionState) -> dict:
    """[T2 평가] 자식 프로그램을 실행해 지표를 측정한다."""
    print("[T2] 평가 중...")

    metrics, feedback = evaluate_program(state["candidate"]["code"])

    print(
        f"--- 지표: correctness={metrics['correctness']:.2f}, calls={metrics['calls']}, "
        f"combined={metrics['combined']} ---"
    )
    for f in feedback:
        print(f"    ✗ {f}")
    return {"candidate": {**state["candidate"], "metrics": metrics}, "feedback": feedback}


def select_node(state: EvolutionState) -> dict:
    """[T3 선택] 자식이 최고 프로그램보다 종합 점수가 높으면 채택(best 갱신), 아니면 기각."""
    candidate = state["candidate"]
    best = state["best"]
    candidate_score = candidate["metrics"]["combined"]
    best_score = best["metrics"]["combined"]
    accepted = candidate_score > best_score

    if accepted:
        print(f"[T3] 채택 ✓  combined {best_score} → {candidate_score} (best 갱신)")
        new_best = candidate
    else:
        if candidate["metrics"]["correctness"] < 1:
            reason = "테스트 실패"
        else:
            reason = "기존 최고보다 효율이 낮음"
        print(f"[T3] 기각 ✗  combined {candidate_score} ≤ {best_score} ({reason})")
        new_best = best

    record = {
        "generation": state["generation"],
        "description": candidate["description"],
        "metrics": candidate["metrics"],
        "accepted": accepted,
    }
    return {"best": new_best, "accepted": accepted, "history": [record]}


def should_continue(state: EvolutionState) -> str:
    """남은 세대가 있으면 변이로 루프백, 아니면 종료."""
    if state["generation"] < MAX_GENERATIONS:
        return "continue"
    return "finish"


def report_node(state: EvolutionState) -> dict:
    """[M2] 최고 프로그램을 결과로 확정한다."""
    return {"best_program": state["best"]}


# ---------------------------------------------------------------------------
# 6. 그래프 구성
#    START → mutate → evaluate → select ─(continue)→ mutate   (루프)
#                                       └(finish)──→ report → END
# ---------------------------------------------------------------------------
graph_builder = StateGraph(EvolutionState)

graph_builder.add_node("mutate", mutate_node)
graph_builder.add_node("evaluate", evaluate_node)
graph_builder.add_node("select", select_node)
graph_builder.add_node("report", report_node)

graph_builder.add_edge(START, "mutate")
graph_builder.add_edge("mutate", "evaluate")
graph_builder.add_edge("evaluate", "select")
graph_builder.add_conditional_edges(
    "select",
    should_continue,
    {
        "continue": "mutate",    # 다음 세대
        "finish": "report",
    },
)
graph_builder.add_edge("report", END)

evolve = graph_builder.compile()


# ---------------------------------------------------------------------------
# 7. 실행
# ---------------------------------------------------------------------------
def print_history(initial_metrics: dict, history: list[dict]) -> None:
    print("\n\n" + "=" * 30 + " 진화 기록 " + "=" * 30)
    print(
        f"세대 0 | 기준   | correctness {initial_metrics['correctness']:.2f} | "
        f"calls {initial_metrics['calls']:>5} | "
        f"combined {initial_metrics['combined']:<5} | {INITIAL_DESCRIPTION}"
    )
    for r in history:
        m = r["metrics"]
        if r["accepted"]:
            status = "채택 ✓"
        else:
            status = "기각 ✗"
        print(
            f"세대 {r['generation']} | {status} | correctness {m['correctness']:.2f} | "
            f"calls {m['calls']:>5} | combined {m['combined']:<5} | {r['description']}"
        )


def print_best_program(best: Program) -> None:
    print("\n" + "=" * 30 + " 최고 프로그램 " + "=" * 30)
    print(f"({best['description']})\n")
    print(best["code"].strip())
    print("\nBest program metrics:")
    for name, value in best["metrics"].items():
        print(f"  {name}: {value}")


def main() -> None:
    print("## 진화형 코딩 에이전트 실행: fib(n) 최적화 ##")

    # 세대 0: 초기 프로그램을 평가해 출발점(best)으로 삼는다.
    metrics, _ = evaluate_program(INITIAL_PROGRAM)
    print(
        f"\n>>> 세대 0: 초기 프로그램  correctness={metrics['correctness']:.2f}, "
        f"calls={metrics['calls']}, combined={metrics['combined']}"
    )
    print("    " + INITIAL_PROGRAM.strip().replace("\n", "\n    "))

    initial_program: Program = {
        "code": INITIAL_PROGRAM,
        "metrics": metrics,
        "description": INITIAL_DESCRIPTION,
    }
    result = evolve.invoke({
        "generation": 0,
        "best": initial_program,
        "feedback": [],
        "history": [],
    })

    print_history(metrics, result["history"])
    print_best_program(result["best_program"])


if __name__ == "__main__":
    main()
