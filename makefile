run: install
	uv run -m src

install:
	uv sync

debug: install
	uv run -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	-find . -name "*.py" -not -path "./llm_sdk/*" -not -path "./.venv/*" -exec flake8 {} +
	python3 -m mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs