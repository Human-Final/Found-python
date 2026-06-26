
# 이 파일의 역할:
# 1. FastAPI 앱 객체를 만든다.
# 2. router.py에서 만든 API 경로를 등록한다.
# 3. 서버 상태 확인용 기본 API를 만든다.


# FastAPI 앱을 만들기 위한 클래스
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ai/router.py에 만든 검색 API 라우터 가져오기
from ai.router import router as search_router


class Utf8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
    

# FastAPI 앱 객체 생성
app = FastAPI(
    version="0.1.0"
)


# 검색어 해석 API 라우터 등록
# router.py에서 prefix="/api/search"를 이미 지정했기 때문에
# 최종 주소는 /api/search/interpret 가 됨
app.include_router(search_router)

# 서버 상태 확인용 API
@app.get("/")
def health_check():

    return {
        "status": "ok",
        "message": "AI Search API is running"
    }