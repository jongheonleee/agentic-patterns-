"""
01. Prompt Chaining 패턴

하나의 큰 작업을 여러 개의 작은 LLM 호출로 쪼개고,
"이전 호출의 출력"을 "다음 호출의 입력"으로 순차적으로 연결하는 패턴.

흐름:
    텍스트 입력 → [체인 1] 기술 사양 추출 → [체인 2] JSON 변환 → 최종 JSON 출력

왜 쓰는가:
    - 한 번에 "추출 + 구조화"를 시키는 것보다 단계를 나누면 각 프롬프트가 단순해져 품질이 안정적이다.
    - 중간 결과를 확인하거나 단계별로 다른 모델/파서를 끼워 넣기 쉽다.

학습용이므로 실제 API 대신 Fake 모델을 사용한다.
FakeListChatModel은 호출될 때마다 responses 리스트의 문자열을 순서대로 돌려준다.
"""
import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# 1. 입력 데이터
# ---------------------------------------------------------------------------
input_text = "새 노트북 모델은 3.5GHz 옥타코어 프로세서, 16GB RAM, 1TB NVMe SSD를 탑재하고 있습니다."


# ---------------------------------------------------------------------------
# 2. Fake 모델 정의
#    체인이 LLM을 두 번 호출하므로 응답도 두 개를 순서대로 준비한다.
#    1번째 응답 → 체인 1(사양 추출)의 결과, 2번째 응답 → 체인 2(JSON 변환)의 결과
# ---------------------------------------------------------------------------
llm = FakeListChatModel(responses=[
    # 체인 1: 사양 추출 결과 (자유 형식 텍스트)
    "CPU: 3.5GHz 옥타코어, RAM: 16GB, Storage: 1TB NVMe SSD",
    # 체인 2: JSON 변환 결과 (JsonOutputParser가 dict로 파싱한다)
    '{"cpu": "3.5GHz 옥타코어", "memory": "16GB", "storage": "1TB NVMe SSD"}'
])


# ---------------------------------------------------------------------------
# 3. 프롬프트 정의
#    {text_input}, {specifications}는 체인 실행 시 채워지는 입력 변수다.
# ---------------------------------------------------------------------------
# 체인 1용: 원문에서 기술 사양만 뽑아내도록 요청
prompt_extract = ChatPromptTemplate.from_template(
    "다음 텍스트에서 기술 사양을 추출해주세요: \n\n{text_input}"
)

# 체인 2용: 추출된 사양을 정해진 키의 JSON으로 구조화하도록 요청
prompt_transform = ChatPromptTemplate.from_template(
    "다음 사양을 'cpu', 'memory', 'storage'를 키로 하는 JSON 객체로 변환해주세요: \n\n{specifications}"
)


# ---------------------------------------------------------------------------
# 4. 체인 구성 (LCEL: `|` 연산자로 Runnable을 이어 붙인다)
# ---------------------------------------------------------------------------
# 체인 1: 프롬프트 → LLM → 문자열 파서
extraction_chain = prompt_extract | llm | StrOutputParser()

# 체인 2를 체인 1 뒤에 연결한 전체 체인
#   {"specifications": extraction_chain}
#     → 딕셔너리 형태로 쓰면 "체인 1의 출력을 specifications 키에 담아라"는 뜻이 된다.
#     → 그 딕셔너리가 prompt_transform의 입력 변수({specifications})로 들어간다.
#   JsonOutputParser()
#     → LLM이 돌려준 JSON 문자열을 Python dict로 변환한다.
full_chain = {"specifications": extraction_chain} | prompt_transform | llm | JsonOutputParser()


# ---------------------------------------------------------------------------
# 5. 실행
#    invoke()에 넘긴 딕셔너리의 키(text_input)는 첫 프롬프트의 변수와 이름이 같아야 한다.
# ---------------------------------------------------------------------------
final_result = full_chain.invoke(
    {"text_input": input_text}
)

print("\n--- 최종 JSON 출력 ---")
print(json.dumps(final_result, indent=2, ensure_ascii=False))
