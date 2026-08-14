.PHONY: install test lint run demo docker-build docker-up

install:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	ruff check src tests

demo:
	python scripts/run_pipeline_demo.py

run:
	uvicorn src.api.main:app --reload

docker-build:
	docker build -t documind-api:local -f infra/docker/Dockerfile .

docker-up:
	docker compose -f infra/docker/docker-compose.yml up --build
