---
title: 'LangChain 완벽 가이드'
layout: page
icon: fas fa-link
permalink: /ai-bootcamp/langchain-guide/
toc: true
tags:
  - LangChain
  - LLM
  - 대형언어모델
  - RAG
  - 체인
  - 에이전트
  - 프롬프트엔지니어링
  - OpenAI
  - 벡터데이터베이스
  - 패스트캠퍼스
  - 패스트캠퍼스AI부트캠프
  - 업스테이지패스트캠퍼스
  - UpstageAILab
  - 국비지원
  - 패스트캠퍼스업스테이지에이아이랩
  - 패스트캠퍼스업스테이지부트캠프
---

# 🔗 LangChain 완벽 가이드

## 📚 개요

LangChain은 대형언어모델(LLM)을 활용한 애플리케이션 개발을 위한 강력한 프레임워크입니다. 복잡한 AI 워크플로우를 체인 형태로 구성하여 더욱 정교하고 실용적인 AI 애플리케이션을 구축할 수 있게 해줍니다.

---

## 🔹 1. LangChain 기초 개념

### 📌 LangChain이란?

LangChain은 LLM을 중심으로 한 애플리케이션 개발 프레임워크로, 다음과 같은 핵심 기능을 제공합니다:

**주요 특징:**
- **체인(Chain)**: 여러 컴포넌트를 연결하여 복잡한 워크플로우 구성
- **프롬프트 템플릿**: 재사용 가능한 프롬프트 관리
- **메모리**: 대화 히스토리 및 상태 관리
- **에이전트**: 도구 활용 및 추론 기반 작업 수행
- **RAG**: 외부 지식 소스와 LLM 연결

### 📌 설치 및 기본 설정

```bash
# Python 설치
pip install langchain
pip install openai
pip install chromadb
pip install tiktoken
```

```python
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# API 키 설정
os.environ['OPENAI_API_KEY'] = 'your-api-key'

# LLM 인스턴스 생성
llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(temperature=0.7)
```

---

## 🔹 2. 핵심 컴포넌트

### 📌 2.1 프롬프트 템플릿 (Prompt Templates)

**기본 프롬프트 템플릿:**
```python
from langchain import PromptTemplate

# 템플릿 정의
template = """
당신은 {role}입니다. 
다음 질문에 {style} 스타일로 답변해주세요:

질문: {question}
답변:
"""

prompt = PromptTemplate(
    input_variables=["role", "style", "question"],
    template=template
)

# 프롬프트 생성
formatted_prompt = prompt.format(
    role="파이썬 전문가",
    style="친근하고 상세하게",
    question="리스트 컴프리헨션이란 무엇인가요?"
)
```

**채팅 프롬프트 템플릿:**
```python
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {domain} 전문가입니다."),
    ("human", "{user_input}")
])

messages = chat_template.format_messages(
    domain="데이터 사이언스",
    user_input="머신러닝과 딥러닝의 차이점을 설명해주세요."
)
```

### 📌 2.2 체인 (Chains)

**기본 체인:**
```python
from langchain.chains import LLMChain

# LLM 체인 생성
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# 체인 실행
result = chain.run({
    "role": "AI 연구자",
    "style": "학술적으로",
    "question": "Transformer 아키텍처의 핵심은 무엇인가요?"
})
```

**순차 체인 (Sequential Chain):**
```python
from langchain.chains import SimpleSequentialChain

# 첫 번째 체인: 주제 요약
summary_template = "다음 텍스트를 한 문장으로 요약하세요: {text}"
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=summary_template
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# 두 번째 체인: 질문 생성
question_template = "다음 요약을 바탕으로 퀴즈 문제를 만드세요: {summary}"
question_prompt = PromptTemplate(
    input_variables=["summary"],
    template=question_template
)
question_chain = LLMChain(llm=llm, prompt=question_prompt)

# 순차 체인 연결
overall_chain = SimpleSequentialChain(
    chains=[summary_chain, question_chain]
)
```

### 📌 2.3 메모리 (Memory)

**대화 버퍼 메모리:**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 메모리 설정
memory = ConversationBufferMemory()

# 대화 체인
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 대화 진행
response1 = conversation.predict(input="안녕하세요! 저는 파이썬을 배우고 있습니다.")
response2 = conversation.predict(input="리스트와 튜플의 차이점을 알려주세요.")
```

**요약 메모리:**
```python
from langchain.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=100
)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)
```

---

## 🔹 3. RAG (Retrieval-Augmented Generation)

### 📌 3.1 문서 로드 및 분할

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 문서 로드
loader = TextLoader('document.txt')
documents = loader.load()

# 문서 분할
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)
```

### 📌 3.2 벡터 저장소 구축

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings()

# 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# 유사도 검색
query = "머신러닝의 정의는 무엇인가요?"
similar_docs = vectorstore.similarity_search(query, k=3)
```

### 📌 3.3 RAG 체인 구축

```python
from langchain.chains import RetrievalQA

# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 질의응답
result = qa_chain("딥러닝과 머신러닝의 차이점을 설명해주세요.")
print(f"답변: {result['result']}")
print(f"출처: {result['source_documents']}")
```

---

## 🔹 4. 에이전트 (Agents)

### 📌 4.1 도구 정의

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import requests

# 커스텀 도구 정의
def search_wikipedia(query: str) -> str:
    """위키피디아에서 정보를 검색합니다."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('extract', '정보를 찾을 수 없습니다.')
    return "검색 실패"

# 도구 등록
tools = [
    Tool(
        name="Wikipedia Search",
        func=search_wikipedia,
        description="위키피디아에서 정보를 검색할 때 사용합니다."
    )
]
```

### 📌 4.2 에이전트 생성 및 실행

```python
# 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 에이전트 실행
response = agent.run("파이썬 프로그래밍 언어에 대해 알려주세요.")
```

---

## 🔹 5. 실제 프로젝트 예시

### 📌 5.1 문서 기반 QA 시스템

```python
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

class DocumentQASystem:
    def __init__(self, pdf_path):
        self.loader = PyPDFLoader(pdf_path)
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.docs = self.text_splitter.split_documents(self.documents)
        
        # 벡터 저장소 구축
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.docs,
            embedding=self.embeddings
        )
        
        # QA 체인 설정
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
    
    def ask_question(self, question):
        return self.qa_chain.run(question)

# 사용 예시
qa_system = DocumentQASystem("research_paper.pdf")
answer = qa_system.ask_question("이 논문의 핵심 기여는 무엇인가요?")
```

### 📌 5.2 대화형 챗봇

```python
from langchain.memory import ConversationBufferWindowMemory

class ConversationalBot:
    def __init__(self):
        self.memory = ConversationBufferWindowMemory(k=5)
        self.llm = ChatOpenAI(temperature=0.7)
        
        # 시스템 프롬프트
        self.system_template = """
        당신은 도움이 되는 AI 어시스턴트입니다.
        이전 대화 내용을 참고하여 일관성 있게 답변해주세요.
        
        이전 대화:
        {chat_history}
        
        현재 질문: {question}
        답변:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=self.system_template
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
    
    def chat(self, message):
        response = self.chain.predict(question=message)
        return response

# 사용 예시
bot = ConversationalBot()
print(bot.chat("안녕하세요! 파이썬을 배우고 싶어요."))
print(bot.chat("어떤 책을 추천해주시나요?"))
```

---

## 🔹 6. 고급 기능 및 최적화

### 📌 6.1 커스텀 체인 개발

```python
from langchain.chains.base import Chain
from typing import Dict, List

class CustomAnalysisChain(Chain):
    """커스텀 분석 체인"""
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    @property
    def input_keys(self) -> List[str]:
        return ["text"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["sentiment", "keywords", "summary"]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        text = inputs["text"]
        
        # 감정 분석
        sentiment_prompt = f"다음 텍스트의 감정을 분석하세요: {text}"
        sentiment = self.llm.predict(sentiment_prompt)
        
        # 키워드 추출
        keyword_prompt = f"다음 텍스트에서 핵심 키워드를 추출하세요: {text}"
        keywords = self.llm.predict(keyword_prompt)
        
        # 요약
        summary_prompt = f"다음 텍스트를 요약하세요: {text}"
        summary = self.llm.predict(summary_prompt)
        
        return {
            "sentiment": sentiment,
            "keywords": keywords,
            "summary": summary
        }
```

### 📌 6.2 성능 최적화

```python
# 캐싱 활용
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

# 스트리밍 응답
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

streaming_llm = ChatOpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 비동기 처리
import asyncio
from langchain.chains import LLMChain

async def async_chain_run():
    chain = LLMChain(llm=llm, prompt=prompt)
    result = await chain.arun({
        "role": "개발자",
        "style": "간단하게",
        "question": "REST API란 무엇인가요?"
    })
    return result
```

---

## 🔹 7. 실습 프로젝트 및 응용

### 📌 7.1 추천 프로젝트

1. **개인 문서 검색 시스템**
   - PDF, 텍스트 파일을 벡터화
   - 자연어 질의로 문서 검색
   - RAG 기반 정확한 답변 생성

2. **업무 자동화 봇**
   - 이메일 요약 및 분류
   - 회의록 생성 및 액션 아이템 추출
   - 보고서 자동 생성

3. **학습 도우미 시스템**
   - 교재 기반 QA 시스템
   - 퀴즈 생성 및 평가
   - 개인화된 학습 계획 수립

### 📌 7.2 디버깅 및 문제해결

**일반적인 문제들:**
```python
# 1. 토큰 길이 초과 문제
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# 2. API 비용 최적화
from langchain.llms import OpenAI

# 더 저렴한 모델 사용
cheap_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# 3. 응답 속도 개선
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
```

---

## 🔹 8. 최신 동향 및 발전 방향

### 📌 8.1 LangChain Expression Language (LCEL)

```python
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# LCEL 체인 구성
chain = (
    prompt
    | llm
    | StrOutputParser()
)

# 병렬 처리
from langchain.schema.runnable import RunnableParallel

parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "translation": translation_chain
})
```

### 📌 8.2 향후 학습 방향

1. **LangSmith**: 프로덕션 모니터링 및 디버깅
2. **LangServe**: API 서버 배포 및 서빙
3. **Multi-modal**: 이미지, 오디오 통합 처리
4. **Fine-tuning**: 도메인 특화 모델 학습 연계

---

## 💡 정리 및 다음 단계

### 🎯 핵심 요점
- **LangChain은 LLM 애플리케이션 개발의 표준 프레임워크**
- **체인, 메모리, 에이전트를 통한 복잡한 워크플로우 구성**
- **RAG를 통한 외부 지식 활용 및 정확도 향상**
- **프로덕션 환경에서의 성능 최적화 필요**

### 📚 추가 학습 자료
- [LangChain 공식 문서](https://docs.langchain.com/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangSmith 플랫폼](https://smith.langchain.com/)

### 🚀 실습 과제
1. 개인 문서 컬렉션으로 RAG 시스템 구축
2. 멀티 에이전트 워크플로우 설계
3. 커스텀 도구 개발 및 통합
4. 프로덕션 배포 및 모니터링 구현

---

*이 가이드는 LangChain 0.1+ 버전을 기준으로 작성되었으며, 지속적으로 업데이트될 예정입니다.*
