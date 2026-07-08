PROJECT_NAME := "stratix"

BRANCH := "$(git branch --show-current)"

PROJECT_DIR := "$(realpath $PWD)"

VERSION := """$(python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")"""

venv := "uv run"
python := "uv run python"

sphinx-build := venv + " sphinx-build --keep-going"
pytest := venv + " pytest"
ruff := venv + " ruff"
mypy := venv + " mypy"
pre-commit := venv + " pre-commit"


# List recipes
list:
    just -l


# Echo information
info:
    @echo {{ PROJECT_NAME }} version {{ VERSION }}, on branch {{ BRANCH }}
    @echo directory {{ PROJECT_DIR }}


# Install the python package locally in editable mode
install:
    {{ python }} -m pip install -e .


# Run all tests with coverage
test:
    {{ pytest }} tests/ --cov=stratix --cov-report=term-missing --cov-report=xml


# Run pre-commit hooks and type checking
lint:
    {{ pre-commit }} run --all-files
    {{ mypy }} src/


# Install pre-commit hooks
lint-install:
    {{ pre-commit }} install


# Build Sphinx HTML docs
docs:
    {{ sphinx-build }} -E docs/ docs/_build/html


# live-reload sphinx docs on file changes
docs-live host="127.0.0.1" port="8001":
    {{venv}} sphinx-autobuild --host {{host}} --port {{port}} \
        --watch src/  \
        --watch examples/  \
        --ignore "docs/examples*" \
        --ignore "*sg_execution_times*" \
        docs/ docs/_build/html

# Clean Sphinx docs
docs-clean:
    rm -rf docs/_build


# Clean generated directories and files
clean:
    rm -rf build/ dist/ *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.pyc' -delete 2>/dev/null || true
    rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage


# Build Python package (sdist + wheel)
package:
    @if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
    @rm -f dist/*
    @{{ python }} -m build --sdist --wheel .


# Build and publish to PyPI
release: package
    @twine upload dist/*
