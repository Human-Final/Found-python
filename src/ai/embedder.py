import os
import json
import numpy as np
import faiss
from openai import OpenAI
from config.setting import settings

class IntentEmbedder:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        
        # 파일 저장 경로 지정 (src/ai/ 폴더 내부)
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(self.current_dir, "intent_index.faiss")
        self.metadata_path = os.path.join(self.current_dir, "metadata.json")
        
        # FAISS 인덱스와 메타데이터 리스트 선언
        self.index = None
        self.metadata = []
        
        # FAISS 인덱스 로드 또는 신규 빌드 가동
        self._load_or_build_vectors()

    def _get_embedding(self, text: str):
        """단일 텍스트를 고차원 임베딩 벡터 배열로 변환"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _load_or_build_vectors(self):
        """로컬에 저장된 FAISS 파일이 있으면 로드하고, 없으면 신규 빌드 후 저장합니다."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("[FAISS] 로컬에서 기존 벡터 DB 인덱스를 감지했습니다. 초고속 로드를 시작합니다...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"[FAISS] 로컬 파일로부터 총 {len(self.metadata)}개 분기 벡터 공간 불러오기 완료. (API 호출 0회)")
        else:
            print("[FAISS] 기존 인덱스 파일이 없습니다. 1,000개 데이터 신규 학습 및 인덱스 빌드를 생성합니다...")
            # 명칭 일치 완료
            self._build_and_save_new_index()

    def _build_and_save_new_index(self):
        """[검색 의도 완벽 교정 패치] '~을 찾아요'가 분실물(found)로 칼같이 가도록 보강합니다."""
        
        # 1. 분실 상황 및 서술형 컨텍스트 풀 대폭 보강 (50개)
        # 1. 분실 상황 및 서술형 컨텍스트 풀 대폭 보강 (75개)
        found_contexts = [
            "지하철에 두고 내린", "버스 정류장에 놔둔", "택시 뒷자리에 흘린", "화장실에 두고 온", "카페 테이블에 깜빡한",
            "길가다 떨어뜨린", "식당 의자에 놔두고 온", "공원 벤치에 두고 간", "독서실 책상에 놔둔", "편의점 카운터에 흘린",
            "술 먹고 잃어버린", "가방 통째로 분실한", "주머니에서 빠진", "쇼핑백 두고 온", "어제 분실한",
            "아까 낮에 잃어버린", "지난주에 분실한", "영화관 좌석에 두고 온", "회사 대기실에 놔둔", "길거리에 흘린",
            "위치는 서울역 부근 날짜는 한달 전", "위치는 강남역 근처 날짜는 어제", "장소는 화장실 부근 시간은 아까 낮에", 
            "위치는 홍대 주변 날짜는 지난주", "날짜는 한달 전 위치는 서울역 부근", "검정색 가방 위치는 서울역 부근",
            "강남역 물품보관함에 넣어두고 깜빡한", "홍대 입구역 계단 아래 떨어뜨린", "버스 하차 태그하다 떨어뜨린",
            "어디선가 흘려서 간절히", "돌려받고 싶어서 애타게", "가방 주우신 분 제발", "소중한 물건인데 깜빡하고",
            "술먹고 정신없이 흘린", "술자리 1차 장소에 두고 온", "노래방 소파 틈새에 흘린", "PC방 모니터 앞에 놔둔",
            "백화점 에스컬레이터에서 떨군", "공항 출국장 로비에 놔두고 내린", "KTX 열차 좌석 그물망에 두고 내린",
            "정류장 의자 밑에 떨어뜨린", "길바닥에 나도 모르게 흘려버린", "자전거 바구니에 깜빡하고 놔둔",
            "산책로 벤치 옆에 떨구고 간", "강의실 맨 뒷자리에 놔두고 그냥 온", "매장 계산대 앞에 두고 온",
            "주차장 차 문 열다가 흘린", "편의점 파라솔 의자에 놔둔", "미용실 대기석에 깜빡하고 둔", "학원 자습실에 놔둔",
            "회사 흡연부스 의자에 깜빡 둔", "은행 창구 서류 작성대 옆에", "한강공원 돗자리 폈던 자리에", "호텔 로비 카운터 옆 소파에",
            "고속버스 짐 칸에 깜빡하고", "광역버스 좌석 앞 그물망 속에", "미술관 관람회장 벤치에", "백화점 푸드코트 정수기 옆에",
            "아파트 분리수거장 앞에 흘린", "단지 내 놀이터 시소 위에 놔둔", "헬스장 개인 사물함 위에 얹어두고 온",
            "수영장 탈의실 헤어드라이기 옆에", "공항 보안검색대 바구니에 깜빡한", "KTX 타러 가다가 대합실 의자에",
            "볼링장 대기석 의자 아래 흘린", "당구장 큐대 거치대 옆에 둔", "포장마차 플라스틱 의자 위에",
            "카페 야외 테라스 테이블에 놔둔", "편의점 전자레인지 위에 놔두고 온", "스터디카페 1인실 책상 아래",
            "미용실 샴푸실 의자 옆에 깜빡한", "약국 조제 대기실 의자에", "주유소 영수증 나오는 곳 옆에",
            "푸드트럭 앞 간이 테이블에 흘린", "서점 베스트셀러 가판대 옆에", "동네 마트 자율포장대 위에 두고 온"
        ]
        
        # 2. 극사실주의 분실 구어체/한탄/변칙 종결형 어근 풀 대폭 보강 (75개)
        found_verbs = [
            "잃어버렸습니다", "분실했습니다", "어디갔지", "사라졌어요", "두고 내림", "놔두고 온 듯", "흘렸나봐요",
            "가져간 사람", "분실신고", "놓치고 감", "깜빡하고 놔둠", "기억이 안나요", "제발 찾아주세요", "잃어버린 사람", 
            "안보임", "안보여요", "어디다 뒀지", "찾아주실 분", "주우신 분 제발 연락주세요", "주우신 분 계신가요", 
            "가져갔나요 돌려받고 싶습니다", "가져갔나요 돌려주세요", "분실했는데 습득된 게 있을까요", "분실물 센터에 습득된 거 있나요", 
            "가져간 사람 천벌받아라", "찾아요", "찾습니다", "찾고 있어요", "가방을 찾아요", "핸드폰 찾아요", "지갑 찾습니다",
            "제발 돌려받고 싶어서 찾아요", "어디선가 흘려서 간절히 찾고 있습니다", "잃어버림", "분실함", "잃어버렸다", 
            "분실했다", "두고옴", "흘림", "잃어버린듯", "분실해버렸음", "분실해버렸다", "어디가버린거지", "어디다처박아둔거지", "돌려받고싶다",
            "사라져버렸음", "기억이안남", "피눈물난다", "멘붕옴", "돌려주세요제발", "증발해버렸네", "누가슬쩍했나",
            "정신머리없이분실", "어디다 떨군거지", "간절하게찾고있음", "도난당한듯", "돌려주면사례함", "사례할테니연락좀",
            "잃어버려서미치겠음", "분실해서환장하겠네", "우짜냐잃어버림", "제발보신분", "찾아주시면보답합니다", "사라져서안보임",
            "돌려주세요피눈물남", "가져간사람돌려줘라", "분실신고완료찾아요", "사라진물건찾음", "두고내려버렸네아놔", "사사라져버렸다",
            "어디간거냐대체", "기억이전혀안남", "정신놓고분실", "잃어버려서눈물남", "제발찾아줘요", "돌려받고싶습니다간절히"
        ]
        
        found_pool = []
        for ctx in found_contexts:
            for verb in found_verbs:
                found_pool.append(f"{ctx} {verb}")
                found_pool.append(f"물건 {ctx} {verb}")
                # 접두어 없이 단독 서술 결합도 풀에 추가하여 숏 쿼리 방어
                found_pool.append(f"{verb}")

        # 1. 습득 상황 및 발견 장소 수식어 컨텍스트 풀 대폭 보강 (75개)
        lost_contexts = [
            "길가다 물건", "바닥에 떨어진 거", "정류장 벤치에 있던 거", "화장실 선반에서", "지하철 의자에 덩그러니",
            "누가 놔두고 간", "카페에 흘리고 간", "주인 잃은 물건", "바닥에 굴러다니던", "주우신 분이 아니라 제가",
            "화장실에서 발견한", "누가 깜빡하고 간 거", "공원에 버려진 듯한", "식당 테이블에 남아있던", "의자에 놓여있던",
            "떨어져 있길래 주운", "계단에 있던 거", "정류장에서 습득한", "잃어버리신 분 찾으라고", "주인 분 보시라고",
            "위치는 서울역 부근 주인을", "위치는 강남역 근처 보관 중", "장소는 화장실 부근 발견한 주인", "쓰레기통 옆에 버려진 듯한",
            "자전거 거치대에 굴러다니던", "홍대입구역 9번출구 앞 바닥에", "택시 내리다가 발판에서 발견한", "편의점 테이블에 덩그러니 있던",
            "독서실 휴게실 의자에 놓여있던", "은행 ATM기 위에 올려져있던", "영화관 좌석 컵홀더에 꽂혀있던", "술집 안주 접시 옆에 있던",
            "주차장 바닥에 떨어져있던거", "지하철 2호선 선반 위에 있던", "버스 맨 뒷자리 시트 사이에 낀", "공항 카트에 놔두고 간",
            "백화점 화장실 세면대에 있던", "공원 잔디밭에 뚝 떨어져있던", "학교 교실 교탁 위에 덩그러니 놓인", "회사 대기실 소파 밑에서",
            "정류장 의자 밑에 굴러다니던", "계단 난간 위에 올려져있던거", "횡단보도 신호등 앞에 떨어져있던", "술 취해서 누가 흘리고 간",
            "정신없이 가다가 떨어뜨리고 간", "앞사람 주머니에서 쏙 빠진 거", "누가 깜빡하고 놓고 간 소중한 물건", "주인 잃고 헤매는 가방",
            "길바닥에 덩그러니 방치된", "아무도 안 가져가서 일단", "비 맞아 가며 길가에 있던 거", "화단 풀숲 사이에 끼어있던",
            "매장 계산대 무인 포스기 옆에", "헬스장 러닝머신 음료 거치대에", "볼링장 테이블 아래에 굴러다니던", "노래방 마이크 옆에 놔두고 간",
            "학원 책상 서랍 속에 든 거", "주차 정산기 위에 올려져있던", "탈의실 사물함 열쇠 꽂힌 채", "주인 없는 폰이 울리길래",
            "바닥에 떨어져서 밟힐 뻔한", "벤치 위에 외롭게 놓여있던", "버 정 의자에 덩그러니", "지하철 환승 통로에서 주운",
            "호텔 로비 의자 사이에 낀", "안내데스크 앞에 떨어져있던", "누가 흘린지 모르게 툭 떨어진", "길바닥 한가운데 굴러다니던",
            "정류장 벤치 구석에 박혀있던", "의자 밑 어두운 곳에 있던", "계단 청소하다 발견한", "화장실 문고리에 걸려있던",
            "누가 가방 통째로 놔두고 간", "바닥에 툭 떨어져서 뒹굴던", "누가 주인을 애타게 찾을 것 같은"
        ]
        
        # 2. 습득 구어체, 감탄사 결합형, 변칙 종결형 어근 풀 대폭 보강 (75개)
        lost_verbs = [
            "주웠습니다", "습득했습니다", "발견했어요", "보관 중입니다", "주인을 찾습니다", "찾아가세요", "맡겨놨어요",
            "떨어져 있네요", "인계 완료", "놓여있음", "주인분 연락주세요", "찾으시는 분 계신가요", "주인 모십니다",
            "가져가세요", "주인 누구임", "사무실에 둠", "경찰서에 줌", "우체통에 넣음", "관리실에 맡김", "주인 애타게 찾을 듯",
            "주웠음", "주움", "줍줍", "득템", "습득함", "발견함", "주웠다", "습득했다", "발견했다", "득템함",
            "아싸 주웠다", "대박 주움", "개이득 주웠음", "와 주웠다", "나이스 득템", "길바닥에서 줍줍", "주워서 보관",
            "아싸 개이득 주움", "나이스 줍줍 완료", "보관하고 있음", "주인 찾아요", "임자 찾습니다", "진짜 주인 연락바람",
            "주인 나타나셈", "가져가라", "주인 누구냐", "경찰서 지구대에 인계함", "인포데스크에 맡겨둠", "카운터에 보관중",
            "관리사무소에 보관중입니다", "우체통에 넣었음", "주인 애타게 찾고있을텐데 연락줘요", "분실자 분 찾아가세요",
            "잃어버리신 분 연락 주세요", "주인 분 보시라고 올려둠", "주인 찾음", "주인 찾습니다 제발", "보관중이니 연락주셈",
            "가져가세요 주인분", "역무실에 맡겨놨습니다", "주인을 기다립니다", "습득물 신고함", "유실물 보관중",
            "습득 완료 연락바람", "주인 꼭 찾으시길", "주우신 분이 아니라 제가 주웠어요", "가져가세요 얼른", "주인분 나타나주세요",
            "잃어버린 주인 분 애타게 찾고 있습니다", "주인 누구인가요", "주인 계십니까", "주인 보시오", "얼른 찾아가세요"
        ]

        lost_pool = []
        for ctx in lost_contexts:
            for verb in lost_verbs:
                lost_pool.append(f"{ctx} {verb}")
                lost_pool.append(f"{ctx} 찾으시는 분 찾음")

        raw_dataset = []
        for txt in list(set(found_pool))[:500]:
            raw_dataset.append({"text": txt, "boardType": "found"})
        for txt in list(set(lost_pool))[:500]:
            raw_dataset.append({"text": txt, "boardType": "lost"})

        texts = [item["text"] for item in raw_dataset]
        self.metadata = [{"text": item["text"], "boardType": item["boardType"]} for item in raw_dataset]
        
        try:
            batch_size = 200
            all_vectors = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                response = self.client.embeddings.create(input=batch_texts, model=self.embedding_model)
                batch_vectors = [data.embedding for data in response.data]
                all_vectors.extend(batch_vectors)
            
            np_vectors = np.array(all_vectors).astype('float32')
            dimension = int(np_vectors.shape[1])
                        
            faiss.normalize_L2(np_vectors)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(np_vectors)
            
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=4)
                
            print(f"[FAISS] 총 {self.index.ntotal}개 문장의 의도 교정 인덱스 빌드 완료!")
            
        except Exception as e:
            print(f"[FAISS 최초 빌드 에러] {e}")


    def predict_board_type(self, user_keyword: str) -> str:
        """사용자 검색어와 가장 문맥이 가까운 boardType을 반환 (임베딩 매칭)"""
        if not user_keyword or user_keyword.strip() == "":
            return "all"

        try:
            user_vector = np.array([self._get_embedding(user_keyword)]).astype('float32')
            faiss.normalize_L2(user_vector)
            
            similarities, indices = self.index.search(user_vector, 1)
            max_sim = float(similarities[0][0])
            best_idx = int(indices[0][0])
            
            if best_idx == -1:
                return "all"
                
            best_match = self.metadata[best_idx]
            
            print(f"[FAISS 의도매칭] 가장 유사한 문장: '{best_match['text']}' | 유사도: {max_sim:.4f}")
            if max_sim < 0.38: 
                return "all"
                
            return best_match["boardType"]
            
        except Exception as e:
            print(f"[FAISS 검색 런타임 에러] {e}")
            return "all"

# 싱글톤 인스턴스 생성
intent_embedder = IntentEmbedder()
