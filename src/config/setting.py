import os
from dotenv import load_dotenv

# 1. 어떤 위치에서 서버를 켜든 무조건 한 단계 상위의 .env를 찾아가도록 경로를 박아버립니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "..", ".env")
load_dotenv(dotenv_path=env_path, override=True)

class Settings:
    def __init__(self):
        # 2. 인스턴스 변수로 호출 시점에 안전하게 가져옵니다.
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
settings = Settings()

# 3. 제대로 가져왔는지 터미널에서 눈으로 직접 확인하는 디버깅 코드
print("========================================")
print("config/setting.py에서 로드한 API KEY:", settings.OPENAI_API_KEY)
print("========================================")
