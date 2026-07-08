import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 시스템 실행 경로 주입으로 모듈 경로 꼬임 원천 차단
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 2. 정방향 교정 완료된 embedder 객체 호출
from ai.embedder import intent_embedder

def run_accuracy_test():
    print("[정확도 테스트] 임베딩 모델 벤치마크 및 검증 연산을 시작합니다...")
    
    # 3. 모델이 한 번도 본 적 없는 "검증용 고난도 테스트 데이터셋" 구축 (정답 트리 지정)
    # 분실자 입장 = found (습득게시판 조회 필요)
    # 습득자 입장 = lost (분실게시판 조회 필요)
    test_dataset = [
        # ----- 복합/도치/고난도 습득자 패턴 (정답: lost) -----
        {"text": "휴대폰 찾으시는 분 찾습니다", "ground_truth": "lost"},
        {"text": "주인 분 애타게 찾고 계실 것 같아서 보관 중입니다", "ground_truth": "lost"},
        {"text": "길가다 지갑 주우신 분이 아니라 제가 주웠으니 주인 찾아요", "ground_truth": "lost"},
        {"text": "에어팟 한쪽 떨어져 있길래 주운 사람입니다", "ground_truth": "lost"},
        {"text": "누가 카페 카운터에 가방 놔두고 가셨어요", "ground_truth": "lost"},
        {"text": "학생증 주워서 우체통에 넣어뒀습니다", "ground_truth": "lost"},
        {"text": "화장실 선반에 갤럭시 워치 발견했습니다 주인분 가져가세요", "ground_truth": "lost"},
        {"text": "자전거 열쇠 꽂혀있길래 일단 관리실에 맡겼습니다", "ground_truth": "lost"},
        {"text": "정류장 벤치에서 서류가방 습득 연락주세요", "ground_truth": "lost"},
        {"text": "주인 분 보시면 댓글 달고 찾아가세요", "ground_truth": "lost"},
        
        # ----- 복합/도치/고난도 분실자 패턴 (정답: found) -----
        {"text": "검정 가방 주우신 분 제발 연락주세요", "ground_truth": "found"},
        {"text": "지하철에 두고 내린 제 소중한 반지 습득하신 분 계신가요", "ground_truth": "found"},
        {"text": "누가 제 핸드폰 가져갔나요 돌려받고 싶습니다", "ground_truth": "found"},
        {"text": "아까 낮에 식당 의자에 파우치 놔두고 온 사람인데 보신 분", "ground_truth": "found"},
        {"text": "어디서 잃어버렸는지 도무지 기억이 안 나요 찾고 있습니다", "ground_truth": "found"},
        {"text": "주머니에 있던 동전 지갑이 언제 빠졌는지 사라짐", "ground_truth": "found"},
        {"text": "노트북 파우치 통째로 분실했는데 습득된 게 있을까요", "ground_truth": "found"},
        {"text": "독서실에 전공서적 두고 왔는데 누가 챙겨갔나요", "ground_truth": "found"},
        {"text": "버스에 목도리 감아두고 그냥 내렸습니다 분실신고 되나요", "ground_truth": "found"},
        {"text": "차키 잃어버려서 차 문을 못 열고 있어요 도와주세요", "ground_truth": "found"}
    ]
    
    results = []
    correct_count = 0
    
    # 4. 루프를 돌며 임베딩 모델의 예측값 추출 및 대조
    for item in test_dataset:
        query = item["text"]
        truth = item["ground_truth"]
        
        # 모델의 예측 가동 ("lost", "found", "all" 중 리턴됨)
        pred = intent_embedder.predict_board_type(query)
        
        is_correct = (pred == truth)
        if is_correct:
            correct_count += 1
            
        results.append({
            "검색어": query,
            "실제정답": truth,
            "모델예측": pred,
            "판정": "정답(O)" if is_correct else "오답(X)"
        })
        
    # 데이터프레임 변환 및 정확도 연산
    df = pd.DataFrame(results)
    total_tests = len(test_dataset)
    accuracy = (correct_count / total_tests) * 100
    
    print(f"📈 테스트 완료! 총 {total_tests}문항 중 {correct_count}문항 적중. 정확도: {accuracy:.2f}%")
    
    # 5. 혼동 행렬(Confusion Matrix) 매트릭스 데이터 집계
    matrix = {"FF": 0, "FL": 0, "LF": 0, "LL": 0, "FA": 0, "LA": 0}
    for r in results:
        t, p = r["실제정답"], r["모델예측"]
        if t == "found" and p == "found": matrix["FF"] += 1
        elif t == "found" and p == "lost": matrix["FL"] += 1
        elif t == "found" and p == "all": matrix["FA"] += 1
        elif t == "lost" and p == "found": matrix["LF"] += 1
        elif t == "lost" and p == "lost": matrix["LL"] += 1
        elif t == "lost" and p == "all": matrix["LA"] += 1

    # 6. Plotly 대시보드 시각화 레이아웃 생성
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🎯 예측 결과 요약', '🧱 의도 혼동 행렬 (Confusion Matrix)'),
        specs=[[{"type": "domain"}, {"type": "xy"}]]
    )
    
    # 좌측: 정답 vs 오답 원형 차트
    fig.add_trace(
        go.Pie(
            labels=['정답 (Correct)', '오답 (Incorrect)'],
            values=[correct_count, total_tests - correct_count],
            marker=dict(colors=['#28a745', '#dc3545']),
            hole=0.4,
            textinfo='value+percent'
        ),
        row=1, col=1
    )
    
    # 우측: 히트맵 대용 바 차트 (정답 트리의 분기 정확성 시각화)
    fig.add_trace(
        go.Bar(name='예측: found', x=['실제: found', '실제: lost'], y=[matrix["FF"], matrix["LF"]], marker_color='#007bff'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name='예측: lost', x=['실제: found', '실제: lost'], y=[matrix["FL"], matrix["LL"]], marker_color='#dc3545'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name='예측: all (판단불가)', x=['실제: found', '실제: lost'], y=[matrix["FA"], matrix["LA"]], marker_color='#6c757d'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'📊 임베딩 모델 분기 분류 정확도 테스트 결과 리포트 (최종 정확도: {accuracy:.1f}%)',
        barmode='stack',
        template='plotly_white'
    )
    
    # 7. 하단에 세부 스코어 테이블 HTML 결합 출력
    html_table = df.to_html(classes='table table-striped', index=False, justify='center')
    
    # 스타일 래핑
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>모델 검증 리포트</title>
        <link rel="stylesheet" href="https://jsdelivr.net">
        <script src="https://plotly.com"></script>
        <style>
            body {{ padding: 30px; font-family: 'Malgun Gothic', sans-serif; background-color: #f8f9fa; }}
            .container {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h2 {{ color: #343a40; margin-bottom: 20px; }}
            .table {{ margin-top: 30px; font-size: 0.95rem; }}
            .th {{ background-color: #e9ecef; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>⚡ 의도 매칭 트리 검증 및 오답 백로그</h2>
            <div id="plotly-chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
            <h3 class="mt-5">📋 세부 쿼리별 채점 결과 기록부</h3>
            {html_table}
        </div>
    </body>
    </html>
    """
    
    output_path = os.path.join(current_dir, "accuracy_test_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("==================================================================")
    print(f"🎉 테스트 대시보드 리포트 추출 완료!")
    print(f"📌 저장 위치: {output_path}")
    print("👉 해당 html 파일을 열어 정확도 수치와 오답 테이블을 정밀 진단해보세요.")
    print("==================================================================")

if __name__ == "__main__":
    run_accuracy_test()
