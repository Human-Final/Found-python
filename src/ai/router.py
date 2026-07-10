# 역할 :
# 1. Spring Boot가 호출할 API 주소를 만든다.
# 2. 요청 JSON을 받는다.
# 3. SearchInterpretService를 호출한다.
# 4. LLM 해석 결과를 JSON으로 반환한다.


# FastAPI에서 API 라우터를 만들기 위한 클래스
# HTTPException은 에러 발생 시 HTTP 상태코드와 메시지를 내려줄 때 사용
from fastapi import APIRouter, HTTPException

# 요청/응답 DTO
from ai.schemas import SearchInterpretRequest, SearchInterpretResponse

# 실제 LLM 호출 서비스
from ai.service import SearchInterpretService
import traceback

# APIRouter 생성
# prefix="/api/search" → 이 라우터 안의 모든 API 주소 앞에 /api/search가 붙음
router = APIRouter(
    prefix="/api/search",
    tags=["search"]
)

# SearchInterpretService 객체 생성
# 여기서 OpenAI 클라이언트도 같이 준비됨
search_service = SearchInterpretService()


# 검색어 해석 API - JSON 형식으로 반환
@router.post("/interpret", response_model=SearchInterpretResponse)
def interpret_search(request: SearchInterpretRequest):

    try:
        # 서비스에 검색어 전달
        return search_service.interpret(request)

    except Exception as e:
        # 예외 발생 시 FastAPI 형식의 에러 응답 반환
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )