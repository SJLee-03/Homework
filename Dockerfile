# 1. 경량화된 Python 공식 Slim 이미지 사용
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /code

# 3. 환경 변수 설정
# PYTHONDONTWRITEBYTECODE: 파이썬이 .pyc 파일을 쓰지 않도록 설정
# PYTHONUNBUFFERED: 파이썬 출력이 버퍼링 없이 즉시 전송되게 하여 로그 확인에 유리
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. 시스템 패키지 설치 및 캐시 정리 (최적화)
# opencv-python-headless를 사용하므로 별도의 무거운 GUI 라이브러리가 불필요합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 종속성 설치 (캐시 레이어 최적화)
# 소스 코드를 복사하기 전에 requirements.txt만 먼저 복사하여
# 소스 코드 변경 시 패키지 재설치를 방지하는 Docker Layer 캐시를 활용합니다.
COPY requirements.txt /code/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 애플리케이션 소스 코드 복사
COPY ./app /code/app

# 7. 보안 강화: root 대신 권한이 제한된 비-루트 유저 생성 및 사용 (DevOps Best Practice)
RUN useradd -m appuser && chown -R appuser:appuser /code
USER appuser

# 8. 컨테이너가 사용할 포트 명시
EXPOSE 8000

# 9. FastAPI 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
