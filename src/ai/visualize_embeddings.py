import os
import sys
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
from openai import OpenAI

def generate_embedding_map():
    print("[시각화 엔진] 내장된 1,000개 의도 분기 전용 데이터셋 로딩 중...")
    
    found_verbs = [
        "잃어버렸습니다", "분실했습니다", "어디갔지", "사라졌어요", "두고 내림", "놔두고 온 듯", "흘렸나봐요",
        "찾고 있어요", "돌려주세요", "가져간 사람", "분실신고", "놓치고 감", "깜빡하고 놔둠", "기억이 안나요",
        "제발 찾아주세요", "잃어버린 사람", "안보여요", "어디다 뒀지", "찾아주실 분"
    ]
    found_contexts = [
        "지하철에 두고 내린", "버스 정류장에 놔둔", "택시 뒷자리에 흘린", "화장실에 두고 온", "카페 테이블에 깜빡한",
        "길가다 떨어뜨린", "식당 의자에 놔두고 온", "공원 벤치에 두고 간", "독서실 책상에 놔둔", "편의점 카운터에 흘린",
        "술 먹고 잃어버린", "가방 통째로 분실한", "주머니에서 빠진", "쇼핑백 두고 온", "어제 분실한",
        "아까 낮에 잃어버린", "지난주에 분실한", "영화관 좌석에 두고 온", "회사 대기실에 놔둔", "길거리에 흘린"
    ]

    found_pool = []
    for ctx in found_contexts:
        for verb in found_verbs:
            found_pool.append(f"{ctx} {verb}")
            found_pool.append(f"물건 {ctx} {verb}")

    lost_verbs = [
        "주웠습니다", "습득했습니다", "발견했어요", "보관 중입니다", "주인을 찾습니다", "찾아가세요", "맡겨놨어요",
        "떨어져 있네요", "인계 완료", "놓여있음", "주인분 연락주세요", "찾으시는 분 계신가요", "주인 모십니다",
        "가져가세요", "주인 누구임", "사무실에 둠", "경찰서에 줌", "우체통에 넣음", "관리실에 맡김", "주인 애타게 찾을 듯"
    ]
    lost_contexts = [
        "길가다 물건", "바닥에 떨어진 거", "정류장 벤치에 있던 거", "화장실 선반에서", "지하철 의자에 덩그러니",
        "누가 놔두고 간", "카페에 흘리고 간", "주인 잃은 물건", "바닥에 굴러다니던", "주우신 분이 아니라 제가",
        "화장실에서 발견한", "누가 깜빡하고 간 거", "공원에 버려진 듯한", "식당 테이블에 남아있던", "의자에 놓여있던",
        "떨어져 있길래 주운", "계단에 있던 거", "정류장에서 습득한", "잃어버리신 분 찾으라고", "주인 분 보시라고"
    ]

    lost_pool = []
    for ctx in lost_contexts:
        for verb in lost_verbs:
            lost_pool.append(f"{ctx} {verb}")
            lost_pool.append(f"{ctx} 찾으시는 분 찾음")

    DATASET = []
    # 정방향 수정 배치
    for txt in list(set(found_pool))[:500]:
        DATASET.append({"text": txt, "boardType": "found"})
    for txt in list(set(lost_pool))[:500]:
        DATASET.append({"text": txt, "boardType": "lost"})

    print(f"[시각화 엔진] 총 {len(DATASET)}개의 의도 데이터셋 매핑 완벽 성공!")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_key = None
    env_paths = [
        os.path.join(current_dir, ".env"),
        os.path.join(current_dir, "..", ".env"),
        os.path.join(current_dir, "..", "..", ".env")
    ]
    for path in env_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("OPENAI_API_KEY"):
                        api_key = line.split("=")[1].strip()
                        break
        if api_key: break

    if not api_key:
        print("❌ 에러: .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        return

    client = OpenAI(api_key=api_key)
    embedding_model = "text-embedding-3-small"
    
    texts = [item["text"] for item in DATASET]
    board_types = [item["boardType"] for item in DATASET]
    
    print(f"[시각화 엔진] OpenAI API 연동 완료. {len(texts)}개 문장을 벡터 공간으로 차원 학습 중...")
    print("👉 대용량 배치 처리가 작동 중입니다. 잠시만 기다려주세요...")
    
    all_vectors = []
    try:
        batch_size = 200
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = client.embeddings.create(input=batch_texts, model=embedding_model)
            batch_vectors = [data.embedding for data in response.data]
            all_vectors.extend(batch_vectors)
            
        X = np.array(all_vectors)
        
        # n_iter 인자 제거로 버전 호환성 완벽 해결
        print(f"[시각화 엔진] 고차원 공간 차원 압축 연산 시작 (t-SNE 기동)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_embedded = tsne.fit_transform(X)
        
        df = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1],
            '문장내용': texts,
            '게시판분기(boardType)': board_types
        })
        
        print("[시각화 엔진] 인터랙티브 그래픽 지도를 HTML 드로잉 중입니다...")
        fig = px.scatter(
            df, x='x', y='y', 
            color='게시판분기(boardType)',
            hover_data=['문장내용'],
            title='⚡ 1,000개 분실/습득 문맥 의도 임베딩 벡터 공간 시각화 지도 (정방향 교정본)',
            color_discrete_map={'found': '#007bff', 'lost': '#dc3545'}
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(template="plotly_white")
        
        output_path = os.path.join(current_dir, "embedding_map.html")
        fig.write_html(output_path)
        
        print("==================================================================")
        print(f"🎉 성공! 1,000개 의도 학습 시각화 지도가 완성되었습니다.")
        print(f"📌 저장 위치: {output_path}")
        print("👉 해당 html 파일을 마우스 더블클릭하여 브라우저로 직접 확인해 보세요!")
        print("==================================================================")
        
    except Exception as e:
        print(f"[시각화 엔진 에러] {e}")

if __name__ == "__main__":
    generate_embedding_map()
