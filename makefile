FUNCTIONS_DEF ?=
INPUT ?=
OUTPUT ?=

run: install
	uv run -m src \
		$(if $(FUNCTIONS_DEF),--functions_definition $(FUNCTIONS_DEF)) \
		$(if $(INPUT),--input $(INPUT)) \
		$(if $(OUTPUT),--output $(OUTPUT))

install:
	uv sync

debug: install
	uv run -m pdb src/__main__.py \
		$(if $(FUNCTIONS_DEF),--functions_definition $(FUNCTIONS_DEF)) \
		$(if $(INPUT),--input $(INPUT)) \
		$(if $(OUTPUT),--output $(OUTPUT))

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	-find . -name "*.py" -not -path "./llm_sdk/*" -not -path "./.venv/*" -exec flake8 {} +
	python3 -m mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs