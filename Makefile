env: 
	uv venv --python 3.12
	uv sync
dataset:
	uv run python3 -m src.get_dataset --set-env

train:
	uv run python src/train.py

format:
	uv run --only-group dev ruff format .
	uv run --only-group dev isort .
	uv run --only-group dev nbqa black notebooks/
	uv run --only-group dev nbqa isort notebooks/

lint:
	uv run --only-group dev ruff check --fix .
