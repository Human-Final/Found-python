
# 역할 
# 1. 프롬프트.py에서 프롬프트를 만듦
# 2. openai api에 프롬프트를 보내고 llm 응답을 json으로 파싱
# 3. searchInterpertResponse 형태로 반환

import json

from openai import OpenAI
from ai.prompter import build_search_prompt     # 함수/변수명은 스네이크 표기법
from ai.schemas import SearchInterpretResponse  # 클래스는 파스칼(카멜과 달리 첫글자도 대문자)
from config.setting import settings

from datetime import datetime, timezone, timedelta

class SearchInterpretService:

    # 서비스 객체가 생성될 때 OpenAI 클라이언트를 만듦
    def __init__(self):     # __init__(self) : 파이썬의 생성자 메서드, 
                            #                  객체가 만들어질 때 자동으로 실행
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    
    # 검색어를 받아서 LLM 해석 결과를 반환
    def interpret(self, keyword: str) -> SearchInterpretResponse:
        
        # 검색어가 비어 있으면 기본값 반환
        if keyword is None or keyword.strip() == "":
            return SearchInterpretResponse()
        
        # LLM에게 보낼 프롬프트 생성
        KST = timezone(timedelta(hours=9))
        today = datetime.now(KST).date().isoformat()
        
        prompt = build_search_prompt(keyword, today)
        
        # OPENAI API 호출
        llm_text = self.call_llm(prompt)
        
        # LLM 응답 문자열을 dict로 변환
        data = self.parse_json(llm_text)
        
        # dict를 응답 DTO로 변환해서 반환
        # ** : 딕셔너리를 풀어서 함수 인자로 넣는 문법
        #      딕셔너리의 키와 값을 매개변수와 매개변수 값으로 변경해주는 것
        # ex) "boardType": "all" -> boardType="all"
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