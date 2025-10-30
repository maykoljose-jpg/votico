.PHONY: ingest run dev clean

ingest:
	python -m app.ingest

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --port 8000

clean:
	rm -rf app/data/index/*
