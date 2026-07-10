
# typing 모듈에서 타입 힌트용 클레스들을 가져옴
# Optional : 값이 있을 수도 있고 none일 수도 있음
from typing import List, Optional
from pydantic import BaseModel


# 요청 DTO
# 자바의 DTO 같은 것. 클라이언트가 API로 보낼 데이터 구조 정의
# 스프링에서 파이썬으로 검색어를 보낼 때 이 구조로 보냄
class SearchInterpretRequest(BaseModel):
    keyword: str                  # 사용자가 입력한 검색어
    boardType: Optional[str] = "all"  # UI에서 선택한 게시판 범위 (all, lost, found)
    startDate: Optional[str] = None  # UI에서 선택한 시작 날짜
    endDate: Optional[str] = None    # UI에서 선택한 종료 날짜
    
# 응답 DTO
# 파이썬 LLM이 검색어를 해석한 결과를 자바로 돌려줌
# 자바의 SearchConditionDTO와 맞춰서 필드명 구성
class SearchInterpretResponse(BaseModel):
    boardType: str = "all"
    status: str = "all"
    category: Optional[str] = None
    color: Optional[str] = None
    place: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    coreKeywords: List[str] = []