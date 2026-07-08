# 필요로 하는 파일(예: service.py) 내부
from ai.embedder import intent_embedder

# 유저 검색어를 넣어 판단 결과 가져오기 ("lost", "found", "all" 중 리턴됨)
board_type_result = intent_embedder.predict_board_type("휴대폰 찾으시는 분")
