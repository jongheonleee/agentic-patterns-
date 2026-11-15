"""
04. Reflection 패턴

LLM이 만든 결과물을 (같은 또는 다른) LLM이 비평하고,
그 피드백을 반영해 다시 생성하는 과정을 반복하여 품질을 끌어올리는 패턴.

흐름:
    작업 요구사항 입력
      → [루프, 최대 N회]
          생성자(Producer): 코드 생성 (2회차부터는 이전 피드백 반영)
          검토자(Reflector): 코드 리뷰 → 피드백 또는 "CODE_IS_PERFECT"
          CODE_IS_PERFECT면 루프 종료, 아니면 피드백을 대화 이력에 추가하고 반복
      → 최종 코드 출력

왜 쓰는가:
    - 한 번에 완벽한 결과를 기대하는 대신, "초안 → 리뷰 → 수정" 사이클을 자동화한다.
    - 검토자에게 별도의 역할(시니어 엔지니어)을 주면 생성자가 놓친 문제를 잡아낼 수 있다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
Fake 모델은 "1차 코드 → 1차 비평 → 2차 코드(개선) → CODE_IS_PERFECT" 순서로 응답하므로
루프가 2회차에서 종료되는 시나리오를 재현한다.
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# 1. Fake 모델 정의
#    생성자와 검토자가 같은 모델을 번갈아 호출하므로, 응답도 "코드 → 비평 → 코드 → 비평" 순으로 번갈아 둔다.
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # [1회차] 생성자: 1차 코드 (docstring·타입 힌트 없음, 재귀 사용)
    """
def calculate_factorial(n):
    if n < 0:
        raise ValueError("음수는 허용되지 않습니다")
    if n == 0:
        return 1
    return n * calculate_factorial(n - 1)
""",
    # [1회차] 검토자: 1차 비평 (개선점 지적)
    """
- docstring이 누락되었습니다
- 타입 힌트가 없습니다
- 재귀 깊이 제한 문제가 있을 수 있습니다
""",
    # [2회차] 생성자: 피드백을 반영한 2차 코드
    '''
def calculate_factorial(n: int) -> int:
    """
    주어진 정수의 팩토리얼을 계산합니다.

    Args:
        n: 팩토리얼을 계산할 음이 아닌 정수

    Returns:
        n의 팩토리얼 값

    Raises:
        ValueError: n이 음수인 경우
    """
    if n < 0:
        raise ValueError("음수는 허용되지 않습니다")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
    # [2회차] 검토자: 더 이상 지적 사항 없음 → 루프 종료 신호
    "CODE_IS_PERFECT"
])


# ---------------------------------------------------------------------------
# 2. 리플렉션 루프
# ---------------------------------------------------------------------------
def run_reflection_loop():
    # 생성자에게 줄 원래 작업. 검토자도 이 요구사항을 기준으로 리뷰한다.
    task_prompt = """
    `calculate_factorial`이라는 Python 함수를 작성하세요.
    이 함수는 다음을 수행해야 합니다:
    - 1. 정수 `n`을 입력으로 받습니다
    - 2. 팩토리얼(n!)을 계산합니다
    - 3. 함수의 기능을 설명하는 명확한 docstring을 포함합니다
    - 4. 엣지 케이스 처리: 0의 팩토리얼은 1입니다
    - 5. 잘못된 입력 처리: 음수가 입력되면 ValueError를 발생시킵니다
    """

    max_iterations = 3   # 무한 루프 방지를 위한 상한. 실제로는 비용/시간 예산에 맞춰 정한다.
    current_code = ""

    # 생성자의 대화 이력. 코드와 피드백이 누적되므로 2회차부터는 "맥락을 아는 상태"로 코드를 개선한다.
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print("\n" + "=" * 25 + f" 리플렉션 루프: 반복 {i+1} " + "=" * 25)

        # ---- 단계 1: 생성 (Producer) ----
        if i == 0:
            print("\n>>> 단계 1: 초기 코드 생성 중...")
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            # 이전 회차의 피드백이 이미 message_history에 들어있으므로, 개선 요청만 덧붙인다.
            print("\n>>> 단계 1: 이전 피드백 기반으로 코드 개선 중...")
            message_history.append(HumanMessage(content="제공된 피드백을 사용하여 코드를 개선해주세요."))
            response = llm.invoke(message_history)
            current_code = response.content

        print(f"\n--- 생성된 코드 (v{i + 1}) ---\n{current_code}")
        message_history.append(response)   # 생성 결과도 이력에 남긴다

        # ---- 단계 2: 검토 (Reflector) ----
        print("\n>>> 단계 2: 생성된 코드 검토 중...")

        # 검토자는 생성자의 대화 이력을 공유하지 않고, 별도의 역할(시니어 엔지니어)로 독립 호출한다.
        # 종료 조건을 명확히 하기 위해 "완벽하면 CODE_IS_PERFECT만 출력"하도록 지시한다.
        reflector_prompt = [
            SystemMessage(content="""
                당신은 시니어 소프트웨어 엔지니어이자 Python 전문가입니다.

                제공된 Python 코드를 원래 작업 요구사항에 따라 꼼꼼하게 코드 리뷰하세요.
                버그, 스타일 문제, 누락된 엣지 케이스, 개선이 필요한 부분을 찾으세요.
                코드가 완벽하고 모든 요구사항을 충족하면 'CODE_IS_PERFECT'라고만 응답하세요.
                그렇지 않으면 피드백을 글머리 기호 목록으로 제공하세요.
            """),
            HumanMessage(content=f"원래 작업:\n{task_prompt}\n\n검토할 코드:\n{current_code}")
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # ---- 단계 3: 종료 판단 ----
        if "CODE_IS_PERFECT" in critique:
            print("\n--- 피드백 ---\n추가 피드백 없음. 코드가 만족스럽습니다.")
            break

        # 피드백을 생성자의 이력에 추가 → 다음 회차에서 이를 반영해 개선한다.
        print(f"\n--- 피드백 ---\n{critique}")
        message_history.append(HumanMessage(content=f"이전 코드에 대한 피드백:\n{critique}"))

    # ---------------------------------------------------------------------------
    # 3. 최종 결과
    #    루프가 CODE_IS_PERFECT로 끝났든, max_iterations에 도달했든 마지막 코드를 출력한다.
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 30 + " 최종 결과 " + "=" * 30)
    print("\n리플렉션 과정 후 최종 코드:\n")
    print(current_code)

if __name__ == "__main__":
    run_reflection_loop()
