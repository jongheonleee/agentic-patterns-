"""
14. Knowledge Retrieval (RAG) 패턴

LLM 은 학습하지 않은 정보(사내 규정, 최신 문서 등)를 알지 못하고, 모르면 그럴듯하게 지어내기도 한다.
RAG(Retrieval-Augmented Generation)는 질문과 관련된 문서를 먼저 검색(Retrieval)하고,
그 문서를 컨텍스트로 LLM 에 함께 넘겨 답변을 생성(Generation)하게 하는 패턴이다.

    색인 (Indexing)   : 문서를 작은 청크로 나누고(Chunking), 각 청크를 벡터로 바꿔(Embedding) 벡터 스토어에 저장한다.
    검색 (Retrieval)  : 질문도 같은 방식으로 벡터로 바꾸고, 코사인 유사도가 높은 청크를 상위 k 개 가져온다.
    필터 (Relevance)  : 유사도가 기준점보다 낮은 청크는 버린다. 관련 문서가 하나도 없으면 "모른다"고 답한다.
    생성 (Generation) : 검색된 청크만 근거로 답변하도록 프롬프트를 구성하고, 답변에 출처를 붙인다.

이 예제는 "가상의 회사 사내 핸드북에 대해 답하는 Q&A 에이전트"다.

흐름:
    M1. 색인 (프로그램 시작 시 1회)
        KNOWLEDGE_BASE 문서 → RecursiveCharacterTextSplitter 로 청킹 → Fake 임베딩 → InMemoryVectorStore
    M2. 사용자 질문 입력
      → [RAG 그래프] R1. retrieve : 질문 임베딩 → 유사도 상위 TOP_K 청크 검색 → 기준점 미만 청크 제거
                    ├─ 관련 청크 있음 → R2. generate  : 청크를 컨텍스트로 프롬프트 구성 → LLM 답변 + 출처
                    └─ 관련 청크 없음 → R3. no_answer : 지어내지 않고 "문서에서 찾지 못했다"고 답변
      → M3. 답변 출력

왜 쓰는가:
    - 모델을 다시 학습시키지 않아도 문서만 바꾸면 최신 정보·사내 정보로 답할 수 있다.
    - 답변의 근거(출처 청크)가 남으므로 사용자가 직접 확인할 수 있다.
    - "검색된 문서에 없으면 모른다고 답하라"는 규칙과 유사도 기준점으로 환각(hallucination)을 줄인다.

08. Memory Management 와의 차이:
    - 08 의 장기 기억은 에이전트가 대화 중에 스스로 쓰고 읽는 "사용자에 대한 기억"이다.
    - RAG 는 사람이 미리 준비한 "지식 문서"를 색인해 두고 읽기만 한다.
      둘 다 임베딩 + 유사도 검색이라는 같은 기술을 쓰지만 목적이 다르다.

원래 스케치와의 대응:
    - 원래 스케치는 웹에서 문서를 내려받고 OpenAIEmbeddings + Weaviate + ChatOpenAI 를 사용했다.
    - 여기서는 API 키·네트워크 없이 실행되도록 다음과 같이 바꿨다.
        문서        : 웹 다운로드        → 코드 안의 KNOWLEDGE_BASE
        임베딩      : OpenAIEmbeddings   → 글자 바이그램 해시 Fake 임베딩 (08 과 같은 방식)
        벡터 스토어 : Weaviate           → langchain_core 의 InMemoryVectorStore
        LLM         : ChatOpenAI         → 컨텍스트에서 문장을 골라 답하는 규칙 기반 Fake 모델
    - retrieve → generate 그래프 구조는 그대로 두고, 관련 문서가 없을 때의 no_answer 분기를 추가했다.
    - 실제 서비스에서는 임베딩·벡터 스토어·LLM 세 곳만 실제 구현으로 바꾸면 된다.

학습용이므로 실제 API 대신 Fake 임베딩과 Fake 모델을 사용한다.
아래 예시 질문을 입력하면 각 경로를 직접 확인할 수 있다.
    1) 연차 휴가는 며칠이야?                 → 휴가 규정 청크 검색 → 답변 + 출처
    2) 재택근무는 일주일에 몇 번 할 수 있어? → 근무 제도 청크 검색 → 답변 + 출처
    3) 노트북 교체 주기가 어떻게 돼?         → 장비 지원 청크 검색 → 답변 + 출처
    4) 회사 근처 맛집 추천해줘               → 관련 청크 없음 → "문서에서 찾지 못했다"
`python 14-knowledge-retrieval-rag.py --demo` 로 실행하면 위 예시를 차례로 자동 실행한다.
"""
import math
import re
import sys
import zlib
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. 설정 · 상수
# ---------------------------------------------------------------------------
# 지식 문서: 실제로는 파일·위키·DB 에서 불러온다. (원래 스케치의 state_of_the_union.txt 역할)
KNOWLEDGE_BASE = [
    {
        "source": "핸드북 1장 - 휴가 규정",
        "text": (
            "입사 첫해에는 매월 개근 시 1일의 연차 휴가가 발생하며, 최대 11일까지 사용할 수 있습니다. "
            "입사 1년이 지나면 매년 15일의 연차 휴가가 주어지고, 3년 이상 근속하면 2년마다 1일씩 늘어납니다. "
            "연차 휴가는 사용 예정일 3일 전까지 인사 시스템에서 신청해야 합니다.\n\n"
            "경조사 휴가는 본인 결혼 5일, 자녀 출산 10일, 부모 상 5일이 주어집니다. "
            "병가는 연 10일까지 유급이며, 3일 이상 연속으로 쓰면 진단서를 제출해야 합니다."
        ),
    },
    {
        "source": "핸드북 2장 - 근무 제도",
        "text": (
            "기본 근무 시간은 하루 8시간이며, 오전 8시에서 10시 사이에 자유롭게 출근할 수 있습니다. "
            "핵심 근무 시간은 오전 10시부터 오후 4시까지로, 이 시간에는 회의와 협업에 참여해야 합니다.\n\n"
            "재택근무는 일주일에 최대 2회까지 가능하며, 전날까지 팀장에게 알려야 합니다. "
            "재택근무 중에도 핵심 근무 시간에는 메신저로 연락이 가능해야 합니다."
        ),
    },
    {
        "source": "핸드북 3장 - 장비 지원",
        "text": (
            "입사자에게는 업무용 노트북 1대와 모니터 1대가 지급됩니다. "
            "노트북 교체 주기는 3년이며, 고장이 나면 IT 지원팀에 요청해 수리하거나 교체받을 수 있습니다.\n\n"
            "재택근무용 장비 구입비는 연 30만 원까지 지원됩니다. "
            "영수증을 첨부해 경비 시스템에 청구하면 다음 달 급여와 함께 지급됩니다."
        ),
    },
    {
        "source": "핸드북 4장 - 보안 수칙",
        "text": (
            "업무용 계정의 비밀번호는 12자 이상이어야 하며 90일마다 변경해야 합니다. "
            "모든 업무 계정에는 2단계 인증을 반드시 설정해야 합니다.\n\n"
            "고객 정보가 담긴 파일은 개인 이메일이나 개인 클라우드로 보내면 안 됩니다. "
            "보안 사고가 의심되면 즉시 보안팀 메신저 채널에 신고합니다."
        ),
    },
    {
        "source": "핸드북 5장 - 교육 · 복지",
        "text": (
            "직무와 관련된 도서와 온라인 강의 비용은 연 100만 원까지 지원됩니다. "
            "외부 컨퍼런스 참가비는 팀장 승인을 받으면 별도로 지원됩니다.\n\n"
            "건강검진은 매년 1회 회사 비용으로 받을 수 있으며, 배우자 검진도 50% 지원됩니다."
        ),
    },
]

# 청킹: 청크가 너무 크면 관련 없는 내용까지 섞이고, 너무 작으면 문맥이 끊긴다.
CHUNK_SIZE = 120       # 청크 하나의 최대 글자 수
CHUNK_OVERLAP = 20     # 앞뒤 청크가 겹치는 글자 수 (문장이 경계에서 잘려도 문맥을 잇기 위해)
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # 문단 → 줄 → 문장 → 단어 순으로 자른다

# 검색
EMBED_DIMS = 1024            # Fake 임베딩 벡터의 차원 수
TOP_K = 3                    # 유사도 상위 몇 개의 청크를 가져올지
RELEVANCE_THRESHOLD = 0.11   # 이 유사도보다 낮은 청크는 관련 없는 것으로 보고 버린다

# Fake 모델이 답변에 쓸 문장 수
ANSWER_SENTENCES = 2

# 생성: {context} 와 {question} 은 체인 실행 시 채워진다.
ANSWER_TEMPLATE = """당신은 회사 사내 핸드북에 대해 답하는 어시스턴트입니다.
아래 [문서] 의 내용만 근거로 [질문] 에 한국어로 간결하게 답하세요.
- 문서에 없는 내용은 추측하지 말고 "문서에서 찾을 수 없습니다"라고 답하세요.
- 답변 끝에 근거로 사용한 문서의 출처를 [출처: ...] 형식으로 붙이세요.

[문서]
{context}

[질문]
{question}

[답변]
"""

NO_ANSWER_MESSAGE = (
    "죄송합니다. 사내 핸드북에서 질문과 관련된 내용을 찾지 못했습니다. "
    "휴가, 근무 제도, 장비, 보안, 교육 · 복지에 대해 질문해 주세요."
)

DEMO_QUESTIONS = [
    "연차 휴가는 며칠이야?",
    "재택근무는 일주일에 몇 번 할 수 있어?",
    "노트북 교체 주기가 어떻게 돼?",
    "회사 근처 맛집 추천해줘",
]


# ---------------------------------------------------------------------------
# 2. 상태(State) 정의
# ---------------------------------------------------------------------------
class RAGGraphState(TypedDict):
    question: str               # 사용자 질문
    documents: list[Document]   # 검색 후 기준점을 넘은 관련 청크
    generation: str             # 최종 답변


# ---------------------------------------------------------------------------
# 3. Fake 임베딩 정의
#    글자 2개 단위(바이그램)를 해시해 고정 차원 벡터에 담는다. (08. Memory Management 와 같은 방식)
#    겹치는 단어가 많을수록 코사인 유사도가 높아진다. 실제로는 OpenAIEmbeddings 등을 넣는다.
# ---------------------------------------------------------------------------
class BigramHashEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        compact = re.sub(r"\s+", "", text)   # 띄어쓰기 차이를 무시한다
        vector = [0.0] * EMBED_DIMS

        for i in range(len(compact) - 1):
            bigram = compact[i:i + 2]
            index = zlib.crc32(bigram.encode("utf-8")) % EMBED_DIMS
            vector[index] += 1.0

        # 길이를 1로 맞춰 두면 내적이 곧 코사인 유사도가 된다.
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# 4. Fake 모델 정의
#    실제 LLM 은 컨텍스트를 읽고 자연어로 답을 쓴다.
#    Fake 모델은 프롬프트의 [문서] 에서 질문과 글자가 가장 많이 겹치는 문장을 골라 출처와 함께 돌려준다.
#    실제 서비스에서는 이 자리에 ChatOpenAI / ChatAnthropic 등을 넣으면 된다.
# ---------------------------------------------------------------------------
class FakeRAGLLM(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-rag-llm"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        prompt = messages[-1].content
        context = prompt.split("[문서]\n", 1)[1].split("\n\n[질문]", 1)[0]
        question = prompt.split("[질문]\n", 1)[1].split("\n\n[답변]", 1)[0]

        question_bigrams = bigrams(question)
        candidates = []   # (점수, 문장, 출처)

        # 컨텍스트는 "[출처: ...]\n본문" 블록이 빈 줄로 이어진 형태다.
        for block in context.split("\n\n"):
            header, _, body = block.partition("\n")
            source = header.removeprefix("[출처: ").removesuffix("]")

            for sentence in re.split(r"(?<=[.!?])\s+", body):
                score = len(bigrams(sentence) & question_bigrams)
                if score > 0:
                    candidates.append((score, sentence.strip(), source))

        if not candidates:
            return "문서에서 찾을 수 없습니다."

        # 질문과 가장 많이 겹치는 문장 ANSWER_SENTENCES 개로 답한다. (중복 문장은 한 번만)
        candidates.sort(key=lambda c: c[0], reverse=True)
        sentences = []
        sources = []
        for _, sentence, source in candidates:
            if sentence in sentences:
                continue
            sentences.append(sentence)
            if source not in sources:
                sources.append(source)
            if len(sentences) == ANSWER_SENTENCES:
                break

        return f"{' '.join(sentences)} [출처: {', '.join(sources)}]"


llm = FakeRAGLLM()


# ---------------------------------------------------------------------------
# 5. 헬퍼 함수
# ---------------------------------------------------------------------------
def bigrams(text: str) -> set[str]:
    """띄어쓰기를 무시한 글자 바이그램 집합. (Fake 모델이 문장을 고를 때 사용)"""
    compact = re.sub(r"\s+", "", text)
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def split_into_chunks(knowledge_base: list[dict]) -> list[Document]:
    """[M1 청킹] 문서를 CHUNK_SIZE 이하의 청크로 나눈다. 출처는 metadata 로 각 청크에 따라간다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        keep_separator="end",   # 마침표는 앞 문장 끝에 붙여 둔다
    )

    texts = [doc["text"] for doc in knowledge_base]
    metadatas = [{"source": doc["source"]} for doc in knowledge_base]
    return splitter.create_documents(texts, metadatas=metadatas)


def format_context(documents: list[Document]) -> str:
    """검색된 청크를 출처와 함께 프롬프트의 {context} 에 넣을 문자열로 만든다."""
    blocks = [
        f"[출처: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in documents
    ]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 6. 색인 (Indexing)
#    청킹 → 임베딩 → 벡터 스토어 저장. 프로그램 시작 시 한 번만 수행한다.
# ---------------------------------------------------------------------------
chunks = split_into_chunks(KNOWLEDGE_BASE)

vectorstore = InMemoryVectorStore(embedding=BigramHashEmbeddings())
vectorstore.add_documents(chunks)

# {context}, {question} 을 채워 LLM 에 넘기고 문자열 답변을 받는 생성 체인
rag_chain = ChatPromptTemplate.from_template(ANSWER_TEMPLATE) | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 7. 노드 정의
# ---------------------------------------------------------------------------
def retrieve_documents_node(state: RAGGraphState) -> dict:
    """[R1] 질문과 유사한 청크를 검색하고, 기준점(RELEVANCE_THRESHOLD) 미만은 버린다."""
    print("\n[R1] retrieve: 유사도 검색")

    results = vectorstore.similarity_search_with_score(state["question"], k=TOP_K)

    relevant_docs = []
    for doc, score in results:
        is_relevant = score >= RELEVANCE_THRESHOLD
        if is_relevant:
            relevant_docs.append(doc)

        mark = "✓" if is_relevant else "✗"
        preview = doc.page_content[:40].replace("\n", " ")
        print(f"    {mark} {score:.3f} | {doc.metadata['source']} | {preview}...")

    print(f"    → 관련 청크 {len(relevant_docs)}개 (기준점 {RELEVANCE_THRESHOLD})")
    return {"documents": relevant_docs}


def generate_response_node(state: RAGGraphState) -> dict:
    """[R2] 검색된 청크만 근거로 답변을 생성한다."""
    print("\n[R2] generate: 검색된 청크로 답변 생성")

    generation = rag_chain.invoke({
        "context": format_context(state["documents"]),
        "question": state["question"],
    })
    return {"generation": generation}


def no_answer_node(_state: RAGGraphState) -> dict:
    """[R3] 관련 문서가 없으면 LLM 을 부르지 않고, 지어내지 않는다는 안내를 돌려준다."""
    print("\n[R3] no_answer: 관련 문서 없음 → 답변을 지어내지 않음")
    return {"generation": NO_ANSWER_MESSAGE}


def route_after_retrieve(state: RAGGraphState) -> str:
    """관련 청크가 하나라도 있으면 생성, 없으면 no_answer 로 보낸다."""
    if state["documents"]:
        return "generate"
    return "no_answer"


# ---------------------------------------------------------------------------
# 8. 그래프 구성
#    START → retrieve ─┬─ (관련 청크 있음) → generate  → END
#                      └─ (관련 청크 없음) → no_answer → END
# ---------------------------------------------------------------------------
graph_builder = StateGraph(RAGGraphState)

graph_builder.add_node("retrieve", retrieve_documents_node)
graph_builder.add_node("generate", generate_response_node)
graph_builder.add_node("no_answer", no_answer_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {
        "generate": "generate",     # 근거가 있을 때만 LLM 으로 답변
        "no_answer": "no_answer",   # 근거가 없으면 모른다고 답변 (환각 방지)
    },
)
graph_builder.add_edge("generate", END)
graph_builder.add_edge("no_answer", END)

rag_agent = graph_builder.compile()


# ---------------------------------------------------------------------------
# 9. 실행
# ---------------------------------------------------------------------------
def print_index_summary() -> None:
    """[M1] 색인 결과: 문서 몇 개가 청크 몇 개로 나뉘었는지 보여준다."""
    print(f"[M1] 색인: 문서 {len(KNOWLEDGE_BASE)}개 → 청크 {len(chunks)}개 "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.page_content[:30].replace("\n", " ")
        print(f"    {i:>2}. {chunk.metadata['source']} | {preview}...")


def handle_question(question: str) -> None:
    print(f"\n[M2] 질문: {question}")

    initial_state = {"question": question, "documents": [], "generation": ""}
    final_state = rag_agent.invoke(initial_state)

    print(f"\n[M3] 답변: {final_state['generation']}")


def main() -> None:
    print("## Knowledge Retrieval (RAG) 사내 핸드북 Q&A 에이전트 실행 ##\n")
    print_index_summary()

    if "--demo" in sys.argv:
        for question in DEMO_QUESTIONS:
            print("\n" + "=" * 80)
            handle_question(question)
        return

    print("\n사내 핸드북에 대해 질문하세요. (종료: q)")
    print("예시:\n    " + "\n    ".join(DEMO_QUESTIONS))

    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):    # Ctrl+D / Ctrl+C 도 오류 없이 종료
            print()
            break

        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue
        handle_question(question)

    print("에이전트를 종료합니다.")


if __name__ == "__main__":
    main()
