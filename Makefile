CACHE = */__pycache__ */.mypy_cache */*.egg-info __pycache__ .mypy_cache *.egg-info
SRC= src
STORAGE := /home/yael-kha/goinfre
VENV_PATH := $(STORAGE)/uv_venv
UV_CACHE_DIR := $(STORAGE)/uv_cache
HF_HOME := $(STORAGE)/HF_cache

export UV_CACHE_DIR
export HF_HOME

install:
	rm -rf .venv
	mkdir -p $(VENV_PATH)
	ln -s $(VENV_PATH) .venv
	uv sync

run: install
	uv run python -m $(SRC)

debug: install
	uv run python3 -m pdb -m $(SRC)

clean:
	rm -rf $(CACHE)
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .venv $(VENV_PATH) $(UV_CACHE_DIR) $(HF_HOME)

lint: install
	uv run --with flake8 python3 -m flake8 $(SRC)
	uv run --with mypy python3 -m mypy -p $(SRC) --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --disable-error-code attr-defined --disable-error-code no-any-return

lint-strict: install
	uv run --with flake8 python3 -m flake8 $(SRC)
	uv run --with mypy python3 -m mypy -p $(SRC) --strict --disable-error-code attr-defined --disable-error-code no-any-return

.PHONY: install run debug clean lint lint-strict