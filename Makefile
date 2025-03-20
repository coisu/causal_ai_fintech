.PHONY: build up down logs shell train test clean reset-db

# 📌 Docker 컨테이너 빌드
build:
	docker-compose build

# 📌 FastAPI + Streamlit + PostgreSQL 실행
up:
	docker-compose up -d

# 📌 FastAPI + Streamlit + PostgreSQL 종료
down:
	docker-compose down

# 📌 로그 확인
logs:
	docker-compose logs -f

# 📌 컨테이너 내부 진입 (FastAPI 서버)
shell:
	docker exec -it causal_ai_app /bin/bash

# 📌 모델 학습 실행
train:
	curl -X GET "http://localhost:8000/train-model"

# 📌 테스트 예측 실행
test:
	curl -X GET "http://localhost:8000/predict-volatility?fed_rate=1.5&unemployment=4.2&cpi=3.1"

# 📌 데이터 및 캐시 정리
clean:
	rm -rf __pycache__ *.pyc *.pyo models/*.pkl

re:
	make down && make clean && make build && make up

# 📌 PostgreSQL 데이터베이스 초기화
reset-db:
	docker-compose down -v && docker-compose up -d db
