# Contributing to stratix

Thank you for considering contributing! Please follow these guidelines.

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/oslumen/stratix
   cd stratix
   ```

2. Create a development environment:
   ```bash
   uv sync --all-extras
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Code Style

- **Formatting**: Ruff formatter (`ruff format .`)
- **Linting**: Ruff linter (`ruff check .`)

- **Type checking**: mypy (`mypy src/`)


Run all linters locally before pushing:

```bash
pre-commit run --all-files
```

## Running Tests

```bash
uv run pytest
```

To run with coverage:

```bash
uv run pytest --cov --cov-report=term-missing
```

## Conventional Commits

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `ci`, `chore`.

## Pull Request Process

1. Create a feature branch off `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass and lint is clean.
4. Update documentation and `CHANGELOG.md` as needed.
5. Open a PR with a clear description and link to related issues.
6. Maintainers will review and merge once checks pass.
