from openai import OpenAI
from .secrets import get_secret

# 이 클라이언트는 모듈이 처음 임포트될 때 한 번만 초기화됩니다.
# Lambda 웜 스타트 시에는 이 초기화 과정 없이 기존 클라이언트 객체가 재사용되어
# Secrets Manager 호출을 방지하고 성능을 최적화합니다.
# Secrets Manager에 {"OPENAI_API_KEY": "sk-..."} 형식으로 저장되어 있다고 가정합니다.
client = OpenAI(api_key = get_secret("OPENAI_SECRET_NAME", "OPENAI_API_KEY"))