# 역할 
# 1. prompter.py에서 프롬프트를 만듦
# 2. openai api에 프롬프트를 보내고 llm 응답을 json으로 파싱
# 3. searchInterpretResponse 형태로 반환
# 4. 하이브리드 고도화: FAISS 임베딩 엔진의 boardType 판정 결과를 우선 융합

import json

from openai import OpenAI
from ai.prompter import build_search_prompt     # 함수/변수명은 스네이크 표기법
from ai.schemas import SearchInterpretResponse,SearchInterpretRequest   # 클래스는 파스칼(카멜과 달리 첫글자도 대문자)
from config.setting import settings

# 고도화 검증 완료된 FAISS 임베딩 인스턴스 가져오기
from ai.embedder import intent_embedder

from datetime import datetime, timezone, timedelta

class SearchInterpretService:

    # 서비스 객체가 생성될 때 OpenAI 클라이언트를 만듦
    def __init__(self):     # __init__(self) : 파이썬의 생성자 메서드, 
                            #                  객체가 만들어질 때 자동으로 실행
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    
    # 검색어를 받아서 LLM 해석 결과를 반환
    def interpret(self, request: SearchInterpretRequest) -> SearchInterpretResponse:
        
        keyword = request.keyword
        
        # 검색어가 비어 있으면 기본값 반환
        if keyword is None or keyword.strip() == "":
            return SearchInterpretResponse()
        
        # [FAISS 의도 분류 가동] 
        # LLM 호출 전, 벡터 공간 상의 문맥 유사도를 비교하여 boardType 분기를 사전 도출합니다.
        embedded_board_type = intent_embedder.predict_board_type(keyword)
        
        # LLM에게 보낼 프롬프트 생성
        KST = timezone(timedelta(hours=9))
        today = datetime.now(KST).date().isoformat()
        
        prompt = build_search_prompt(keyword, today)
        
        # OPENAI API 호출
        llm_text = self.call_llm(prompt)
        
        # LLM 응답 문자열을 dict로 변환
        data = self.parse_json(llm_text)
        
        # [하이브리드 병합 정책 적용 - 1단계: FAISS 반영]
        # LLM이 간혹 헷갈리는 문맥적 boardType 결과 대신,
        # 우리가 90% 정확도로 튜닝한 FAISS 결과가 'all'이 아니라면 그 값을 최종 데이터로 채택합니다.
        if embedded_board_type != "all":
            print(f"[의도 고도화 반영] boardType을 FAISS 검증 결과로 업데이트합니다: {embedded_board_type}")
            data["boardType"] = embedded_board_type
            
        # [하이브리드 병합 정책 적용 - 2단계: UI 선택값 최우선 덮어쓰기] 🚀 추가된 구역
        # 사용자가 UI에서 '전체(all)'가 아닌 'lost'나 'found'를 명시적으로 선택했다면,
        # LLM과 FAISS 결과를 모두 무시하고 UI 선택값을 최종 채택합니다.
        if request.boardType and request.boardType != "all":
            print(f"[UI 우선순위 반영] 사용자가 화면에서 선택한 boardType으로 강제 변경합니다: {request.boardType}")
            data["boardType"] = request.boardType
            
        # 🚀 [수정] 사용자가 UI에서 날짜를 지정했다면 LLM 추론본을 완전 무시하고 무조건 강제 주입
        if request.startDate and request.startDate.strip() != "":
            print(f"[UI 우선순위 반영] startDate 강제 주입: {request.startDate}")
            data["startDate"] = request.startDate
        else:
            data["startDate"] = None # 명시적으로 세팅하여 유실 방지
            
        if request.endDate and request.endDate.strip() != "":
            print(f"[UI 우선순위 반영] endDate 강제 주입: {request.endDate}")
            data["endDate"] = request.endDate
        else:
            data["endDate"] = None
        
        # dict를 응답 DTO로 변환해서 반환
        return SearchInterpretResponse(**data)
    
    # AI 호출
    def call_llm(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "너는 분실물/습득물 검색어를 JSON으로만 구조화하는 API다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,      # 구조화 목적이기 때문에 0으로 설정
            response_format={"type": "json_object"}
        )

        # LLM 응답 본문만 꺼내서 반환
        return response.choices[0].message.content

    # LLM 응답 문자열을 Python dict로 변환
    def parse_json(self, text: str) -> dict:

        if text is None or text.strip() == "":
            raise ValueError("LLM 응답이 비어 있습니다.")

        text = text.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            # JSON 앞뒤에 문장이 붙은 경우 대비
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or start >= end:
                raise ValueError(f"LLM 응답에서 JSON을 찾을 수 없습니다: {text}")

            json_text = text[start:end + 1]

            return json.loads(json_text)
