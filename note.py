import os 
import json 

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

input_text = "새 노트북 모델은 3.5GHz 옥타코어 프로세서, 16GB RAM, 1TB NVMe SSD를 탑재하고 있습니다."

llm = FakeListChatModel(
    responses = [
        "CPU: 3.5GHz 옥타코어, RAM: 16GB, Storage: 1TB NVMe SSD",
        '{"cpu": "3.5GHz 옥타코어", "memory": "16GB", "storage": "1TB NVMe SSD"}'
    ]
)

prompt_extract = ChatPromptTemplate.from_template(
    "다음 텍스트에서 기술 사양을 추출하세요: \n\n{text_input}"
)

prompt_transform = ChatPromptTemplate.from_template(
    "다음 사양을 'cpu', 'memory', 'storage'를 키로 하는 JSON 객체로 변환해주세요: \n\n{specifications}"
)

extraction_chain = prompt_extract | llm | StrOutputParser()
full_chain = {"specifications": extraction_chain} | prompt_transform | llm | JsonOutputParser()

final_result = full_chain.invoke(
    {"text_input": input_text}
)

print("\n--- Final JSON Output ---")
print(json.dumps(final_result, indent=2, ensure_ascii=False))
