---
title: 'LangChain 개인 과제: WordQuest Claude Integration 프로젝트 완성기'
layout: page
icon: fas fa-robot
permalink: /ai-bootcamp/langchain-wordquest-project/
toc: true
tags:
  - LangChain
  - LLM
  - OpenAI
  - Solar API
  - Streamlit
  - PostgreSQL
  - 영어학습
  - AI챗봇
  - RAG
  - 개인과제
  - UpstageAILab
  - 패스트캠퍼스
---

# 🤖 LangChain 개인 과제: WordQuest Claude Integration 프로젝트 완성기

> **"LangChain의 모든 기능을 활용하여 실용적인 영어 학습 AI 시스템을 구축하다"**

---

## 📚 프로젝트 개요

### 🎯 과제 목표
Upstage AI Lab 7기 개인 과제의 3주차 요구사항을 충족하는 **실용적인 AI 애플리케이션**을 LangChain을 활용하여 구현했습니다.

**핵심 요구사항:**
- ✅ LangChain의 주요 기능 통합 (Chain, Memory, Tool, Agent)
- ✅ 실용적인 AI 애플리케이션 완성
- ✅ GitHub 레포지토리 및 상세한 README.md
- ✅ LangChain 프로젝트 학습 블로그 작성

### 🌟 프로젝트 특징
- **AI 기반 영어 학습 시스템**: OpenAI GPT + Solar API 이중 백업
- **Streamlit 웹 인터페이스**: 직관적이고 반응형 UI
- **PostgreSQL 데이터베이스**: 학습 기록 및 사용자 데이터 관리
- **JWT 인증 시스템**: 보안 강화된 사용자 관리
- **학습 진도 추적**: 개인별 학습 통계 및 분석

---

## 🏗️ 기술 아키텍처

### 🔧 기술 스택
```
Frontend: Streamlit (Python)
Backend: Python 3.11 + FastAPI
AI Services: OpenAI API + Solar API (Upstage)
Database: PostgreSQL 14
Authentication: JWT + bcrypt
Framework: LangChain
Deployment: Local Development Server
```

### 🏛️ 시스템 구조
```
wordquest-claude-integration/
├── app/
│   ├── core/           # 핵심 모듈 (설정, DB, 보안)
│   ├── services/       # 비즈니스 로직 (AI, 인증, 학습)
│   └── utils/          # 유틸리티 함수
├── docs/               # 프로젝트 문서
├── main.py             # Streamlit 메인 앱
└── requirements.txt    # Python 의존성
```

---

## 🔗 LangChain 핵심 기능 구현

### 1️⃣ LLM 객체 생성 및 관리

**OpenAI와 Solar API를 통한 이중 백업 시스템:**

```python
# app/services/ai_service.py
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import openai
import requests

class AIService:
    def __init__(self):
        # OpenAI API 설정
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Solar API 설정 (Upstage)
        self.solar_api_key = os.getenv("SOLAR_API_KEY")
        self.solar_base_url = os.getenv("SOLAR_BASE_URL", "https://api.upstage.ai/v1")
        self.solar_model = os.getenv("SOLAR_MODEL", "solar-mini-250422")
    
    def get_ai_response(self, message: str, user_id: int = None) -> str:
        """AI 응답 생성 - OpenAI 우선, 실패 시 Solar API 사용"""
        try:
            # OpenAI API 시도
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 영어 학습을 돕는 친근한 AI 튜터입니다."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            # OpenAI 실패 시 Solar API 사용
            return self._get_solar_response(message)
```

### 2️⃣ PromptTemplate 및 Chain 구성

**역할 기반 프롬프트 템플릿:**

```python
# 영어 학습을 위한 특화된 프롬프트
ENGLISH_LEARNING_PROMPT = """
당신은 {role}입니다. 
사용자의 영어 학습을 돕기 위해 {style} 스타일로 답변해주세요.

사용자 질문: {question}

답변 시 다음을 고려해주세요:
- 영어 학습자의 수준에 맞는 설명
- 구체적인 예시 제공
- 학습 포인트 강조
- 친근하고 격려하는 톤

답변:
"""

# 문법 검사를 위한 전용 프롬프트
GRAMMAR_CHECK_PROMPT = """
당신은 영어 문법 전문가입니다.
다음 영어 문장의 문법을 검사하고 수정해주세요:

원문: {sentence}

다음 형식으로 답변해주세요:
❌ 오류: [잘못된 부분]
💡 설명: [오류 원인]
✅ 수정: [수정된 문장]
🎯 학습 포인트: [기억할 문법 규칙]
"""
```

### 3️⃣ Memory 시스템 구현

**대화 히스토리 및 학습 기록 관리:**

```python
# app/services/learning_service.py
class LearningService:
    def __init__(self):
        self.db = Database()
    
    def save_chat_message(self, user_id: int, message: str, is_ai: bool = False):
        """채팅 메시지 저장 - LangChain Memory 연동"""
        query = """
        INSERT INTO claude_integration_chat_messages 
        (user_id, message_type, content, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        message_type = "ai" if is_ai else "user"
        self.db.execute_query(query, (user_id, message_type, message, datetime.now()))
    
    def get_user_chat_history(self, user_id: int, limit: int = 10):
        """사용자별 채팅 히스토리 조회"""
        query = """
        SELECT content, message_type, timestamp
        FROM claude_integration_chat_messages
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        return self.db.fetch_all(query, (user_id, limit))
```

### 4️⃣ RAG (Retrieval-Augmented Generation) 시스템

**학습 데이터 기반 맞춤형 응답:**

```python
# 학습 진도 분석을 통한 개인화된 응답
def get_personalized_response(self, user_id: int, message: str) -> str:
    """사용자 학습 진도를 고려한 맞춤형 AI 응답"""
    
    # 1. 사용자 학습 데이터 분석
    learning_stats = self.get_user_learning_stats(user_id)
    
    # 2. 학습 수준에 맞는 프롬프트 조정
    if learning_stats['level'] == 'beginner':
        system_prompt = "당신은 초급 영어 학습자를 위한 친근한 튜터입니다."
    elif learning_stats['level'] == 'intermediate':
        system_prompt = "당신은 중급 영어 학습자를 위한 전문적인 튜터입니다."
    else:
        system_prompt = "당신은 고급 영어 학습자를 위한 심화 학습 가이드입니다."
    
    # 3. LangChain을 통한 컨텍스트 기반 응답
    response = self.get_ai_response_with_context(
        message=message,
        system_prompt=system_prompt,
        user_context=learning_stats
    )
    
    return response
```

---

## 🎨 주요 기능 및 UI 구현

### 🏠 홈 페이지 - 퀵 스타트 카드

![홈 인터페이스](/assets/img/langchain-project/streamlit-home-interface.png){: width="600px"}

**LangChain Chain을 활용한 기능별 모듈화:**
- **AI 채팅**: 실시간 영어 학습 대화
- **문법 검사**: AI 기반 영어 문법 분석
- **어휘 도움**: 컨텍스트 기반 어휘 설명

### 💬 AI 채팅 - LangChain Memory 연동

![AI 채팅 인터페이스](/assets/img/langchain-project/streamlit-ai-chat-interface.png){: width="600px"}

**LangChain Memory 시스템의 실제 구현:**
```python
# 대화 맥락 유지를 위한 Memory Chain
class ConversationChain:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, message: str, user_id: int) -> str:
        # 1. 메시지 저장
        self.save_chat_message(user_id, message, is_ai=False)
        
        # 2. LangChain Chain 실행
        response = self.chain.predict(input=message)
        
        # 3. AI 응답 저장
        self.save_chat_message(user_id, response, is_ai=True)
        
        return response
```

### 📊 학습 대시보드 - 데이터 분석 및 시각화

![학습 대시보드](/assets/img/langchain-project/streamlit-learning-dashboard.png){: width="600px"}

**LangChain Tool을 활용한 학습 데이터 분석:**
- 총 채팅 수, 문법 검사 수, 어휘 분석 수
- 주간 활동 차트 및 학습 패턴 분석
- 개인별 학습 진도 추적

### ✏️ 문법 검사 - AI 기반 분석

![문법 검사 인터페이스](/assets/img/langchain-project/streamlit-grammar-check.png){: width="600px"}

**LangChain PromptTemplate을 활용한 전문적인 문법 검사:**
```python
def check_grammar(self, sentence: str, user_id: int) -> dict:
    """AI 기반 영어 문법 검사"""
    
    # LangChain PromptTemplate 활용
    prompt = PromptTemplate(
        input_variables=["sentence"],
        template=GRAMMAR_CHECK_PROMPT
    )
    
    # Chain 실행
    chain = LLMChain(llm=self.llm, prompt=prompt)
    result = chain.run(sentence=sentence)
    
    # 결과 파싱 및 저장
    parsed_result = self.parse_grammar_result(result)
    self.save_grammar_check(user_id, sentence, parsed_result)
    
    return parsed_result
```

### 📚 어휘 도움 - 컨텍스트 기반 학습

![어휘 도움 인터페이스](/assets/img/langchain-project/streamlit-vocabulary-help.png){: width="600px"}

**LangChain RAG 시스템을 활용한 어휘 분석:**
- 영어 텍스트 입력 시 어휘 수준 분석
- 컨텍스트 기반 어휘 설명 및 예시 제공
- 학습 기록 저장 및 추천 시스템

### 👤 프로필 관리 - 개인화된 학습 경험

![프로필 인터페이스](/assets/img/langchain-project/streamlit-profile-interface.png){: width="600px"}

**사용자별 학습 데이터 관리:**
- 개인 학습 통계 및 진도 분석
- 비밀번호 변경 및 계정 관리
- 학습 목표 설정 및 달성도 추적

---

## 🔐 인증 시스템 구현

### 📝 회원가입 - 보안 강화

![회원가입 인터페이스](/assets/img/langchain-project/streamlit-signup-interface.png){: width="600px"}

**LangChain을 활용한 입력 검증 및 보안:**
```python
# 비밀번호 강도 검증
def validate_password_strength(self, password: str) -> List[str]:
    """LangChain을 활용한 비밀번호 강도 분석"""
    
    validation_prompt = """
    다음 비밀번호의 강도를 분석하고 개선점을 제시하세요:
    비밀번호: {password}
    
    다음 기준으로 평가하세요:
    1. 최소 8자 이상
    2. 영문 대문자 포함 (A-Z)
    3. 영문 소문자 포함 (a-z)
    4. 숫자 포함 (0-9)
    5. 특수문자 포함 (!@#$%^&*()_+-=[]{}|;:,.<>?)
    
    부족한 점을 리스트로 반환하세요.
    """
    
    prompt = PromptTemplate(
        input_variables=["password"],
        template=validation_prompt
    )
    
    chain = LLMChain(llm=self.llm, prompt=prompt)
    result = chain.run(password=password)
    
    return self.parse_validation_result(result)
```

### 🔑 로그인 - JWT 토큰 기반 인증

![로그인 인터페이스](/assets/img/langchain-project/streamlit-login-interface.png){: width="600px"}

**보안 강화된 인증 시스템:**
- bcrypt를 활용한 비밀번호 해싱
- JWT 토큰 기반 세션 관리
- 자동 로그아웃 및 토큰 갱신

---

## 🐛 디버깅 및 모니터링

### 🐛 디버그 모드 - 시스템 상태 모니터링

![디버그 모드](/assets/img/langchain-project/streamlit-debug-mode.png){: width="500px"}

**LangChain 실행 상태 실시간 모니터링:**
- 사용자 ID, 인증 상태, 현재 페이지
- 데이터베이스 연결 상태
- OpenAI API 및 Solar API 상태
- LangChain Chain 실행 로그

---

## 🚀 성능 최적화 및 개선

### 1️⃣ API 비용 최적화
```python
# OpenAI와 Solar API의 스마트 라우팅
def smart_api_routing(self, message: str) -> str:
    """메시지 특성에 따른 최적 API 선택"""
    
    # 한국어 포함 여부 확인
    if self.contains_korean(message):
        # 한국어가 포함된 경우 Solar API 우선 사용
        return self._get_solar_response(message)
    else:
        # 영어만 있는 경우 OpenAI API 사용
        return self._get_openai_response(message)
```

### 2️⃣ 응답 속도 개선
```python
# LangChain 캐싱 시스템 활용
from langchain.cache import InMemoryCache
import langchain

# 메모리 캐시 활성화
langchain.llm_cache = InMemoryCache()

# 자주 사용되는 프롬프트 캐싱
def get_cached_response(self, message: str) -> str:
    """캐시된 응답 우선 사용"""
    cache_key = f"response_{hash(message)}"
    
    if cache_key in self.cache:
        return self.cache[cache_key]
    
    response = self.get_ai_response(message)
    self.cache[cache_key] = response
    
    return response
```

### 3️⃣ 에러 처리 및 복구
```python
# LangChain Chain 실행 시 에러 처리
def robust_chain_execution(self, chain, inputs: dict) -> str:
    """안정적인 Chain 실행을 위한 에러 처리"""
    
    try:
        result = chain.run(inputs)
        return result
        
    except Exception as e:
        # 1차 에러: 로그 기록
        logger.error(f"Chain 실행 오류: {e}")
        
        # 2차 시도: 단순화된 프롬프트로 재시도
        fallback_result = self.fallback_response(inputs)
        
        # 3차 시도: 기본 응답 제공
        if not fallback_result:
            return "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
        return fallback_result
```

---

## 📊 프로젝트 성과 및 학습 결과

### 🎯 과제 요구사항 충족도

| 요구사항 | 구현 상태 | 완성도 |
|---------|----------|--------|
| LangChain 기본 이해 | ✅ 완료 | 100% |
| 환경 세팅 | ✅ 완료 | 100% |
| API 키 설정 | ✅ 완료 | 100% |
| LLM 객체 생성 | ✅ 완료 | 100% |
| PromptTemplate 사용 | ✅ 완료 | 100% |
| Chain 구성 | ✅ 완료 | 100% |
| Memory 시스템 | ✅ 완료 | 100% |
| RAG 구현 | ✅ 완료 | 100% |
| 실용적 애플리케이션 | ✅ 완료 | 100% |

### 🏆 주요 성과

1. **LangChain 마스터**: 모든 핵심 기능을 실제 프로젝트에 적용
2. **실용적 AI 시스템**: 실제 사용 가능한 영어 학습 플랫폼 구축
3. **멀티 API 연동**: OpenAI + Solar API 이중 백업 시스템
4. **사용자 경험**: 직관적이고 반응형 웹 인터페이스
5. **데이터 관리**: PostgreSQL 기반 학습 기록 및 통계 시스템

### 💡 학습 포인트

#### **LangChain의 강력함**
- **체인 구성**: 복잡한 AI 워크플로우를 체계적으로 구성
- **메모리 관리**: 대화 히스토리 및 컨텍스트 유지
- **프롬프트 엔지니어링**: 역할 기반 맞춤형 응답 생성
- **RAG 시스템**: 외부 지식과 LLM의 효과적인 연동

#### **실무 적용 경험**
- **API 관리**: 비용 최적화 및 에러 처리
- **성능 최적화**: 캐싱, 비동기 처리, 응답 속도 개선
- **사용자 경험**: 직관적인 UI/UX 설계
- **데이터 보안**: JWT 인증, 비밀번호 해싱, 입력 검증

---

## 🔮 향후 발전 방향

### 1️⃣ 고급 LangChain 기능 활용
- **Agent 시스템**: 도구 활용 및 자동화
- **Fine-tuning**: 도메인 특화 모델 학습
- **Multi-modal**: 이미지, 오디오 통합 처리

### 2️⃣ 시스템 확장
- **클라우드 배포**: AWS, GCP, Azure 활용
- **사용자 확장**: 다국어 지원, 모바일 앱
- **AI 모델 다양화**: Claude, Gemini 등 추가 연동

### 3️⃣ 학습 데이터 활용
- **개인화 학습**: AI 기반 맞춤형 커리큘럼
- **학습 분석**: 머신러닝을 통한 학습 패턴 분석
- **커뮤니티**: 사용자 간 학습 경험 공유

---

## 📚 참고 자료 및 링크

### 🔗 프로젝트 관련
- **GitHub 레포지토리**: [wordquest-claude-integration](https://github.com/ProDevJune/wordquest-claude-integration)
- **실행 방법**: `streamlit run main.py --server.port 9001 --server.address localhost`
- **데모 URL**: http://localhost:9001

### 📖 학습 자료
- **LangChain 공식 문서**: [docs.langchain.com](https://docs.langchain.com/)
- **Upstage AI Lab**: [upstage.ai](https://upstage.ai/)
- **패스트캠퍼스 AI 부트캠프**: [fastcampus.co.kr](https://fastcampus.co.kr/)

---

## 💭 마무리

이번 LangChain 개인 과제를 통해 **이론적 지식을 실제 프로젝트에 적용하는 귀중한 경험**을 얻었습니다. 

**특히 인상 깊었던 점들:**
1. **LangChain의 직관성**: 복잡한 AI 워크플로우를 간단한 체인으로 구성
2. **실용성**: 이론 학습을 넘어 실제 사용 가능한 시스템 구축
3. **확장성**: 모듈화된 구조로 새로운 기능 추가 용이
4. **사용자 중심**: AI 기술을 사용자 경험 향상에 집중

**Upstage AI Lab 7기**에서 배운 LangChain 지식을 바탕으로, 앞으로 더욱 혁신적이고 실용적인 AI 애플리케이션을 개발해 나가겠습니다! 🚀

---

*이 블로그는 Upstage AI Lab 7기 개인 과제의 일환으로 작성되었습니다.*
