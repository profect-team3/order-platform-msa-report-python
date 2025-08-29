import os
import json
import boto3
import logging

logger = logging.getLogger(__name__)

_cached_secrets = {}  # 모듈 레벨 캐시

def get_secret(secret_name_env_var: str, secret_json_key: str) -> str:
    """
    AWS Secrets Manager에서 시크릿을 안전하게 가져옵니다.

    결과는 메모리에 캐싱되어 Lambda 실행 컨테이너가 재사용될 때
    불필요한 API 호출을 방지합니다.

    Args:
        secret_name_env_var: 시크릿의 이름(또는 ARN)을 담고 있는 환경 변수의 이름.
        secret_json_key: 시크릿 값(JSON) 내에서 원하는 값의 키.

    Returns:
        요청된 시크릿 값.
    
    Raises:
        ValueError: 필요한 환경 변수가 설정되지 않았을 경우.
        KeyError: 시크릿 JSON에 요청한 키가 없을 경우.
    """
    cache_key = f"{secret_name_env_var}:{secret_json_key}"
    if cache_key in _cached_secrets:
        return _cached_secrets[cache_key]

    secret_name = os.environ.get(secret_name_env_var)
    if not secret_name:
        raise ValueError(f"'{secret_name_env_var}' 환경 변수가 설정되지 않았습니다.")

    region_name = os.environ.get("AWS_REGION", "ap-northeast-2")
    
    logger.info("Fetching secret '%s' from Secrets Manager in region %s", secret_name, region_name)
    
    session = boto3.session.Session()
    sm_client = session.client(service_name='secretsmanager', region_name=region_name)

    get_secret_value_response = sm_client.get_secret_value(SecretId=secret_name)
    secret_string = get_secret_value_response['SecretString']
    secret_dict = json.loads(secret_string)
    
    secret_value = secret_dict[secret_json_key]
    _cached_secrets[cache_key] = secret_value
    return secret_value