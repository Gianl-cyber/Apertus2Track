# Hack Apertus Track 2A - See It, Say It, Pick It
# Everything runs in Docker: see docker-compose.yml and README.md.
# The answers are written to outputs/results_heldout.json.

.PHONY: run perception decision test

# One step: both models on the same GPU (>= 40 GB)
run:
	docker compose run --rm --build pipeline
	docker compose down

# Two steps for a 12 GB GPU: one model at a time.
# Step 1: Qwen2.5-VL writes the scene graphs to outputs/heldout/
perception:
	docker compose up -d --wait ollama
	docker compose run --rm --build --no-deps pipeline python run.py --data data/heldout --perception-only
	docker compose down

# Step 2: Apertus decides and plans from the saved scene graphs
decision:
	docker compose up -d --wait apertus
	docker compose run --rm --build --no-deps pipeline python run.py --data data/heldout --out outputs/results_heldout.json --reuse-perception
	docker compose down

# Unit tests, no model needed
test:
	docker compose run --rm --build --no-deps pipeline python -m unittest discover -s tests
