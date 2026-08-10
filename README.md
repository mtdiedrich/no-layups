# no-layups

## Quick Start

```powershell
# Create virtual environment
uv venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -e .[dev]

# Run commands
uv run pytest -v                                    # Run tests
uv run pytest --cov=no_layups --cov-report=html         # Tests with coverage
uv run ruff check .                                 # Check code quality
uv run ruff format .                                # Format code
```