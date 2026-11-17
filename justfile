# Developer shortcuts (https://github.com/casey/just). Optional: every recipe
# is a plain `uv run` command that CI and the docs also call directly.
# Each recipe line runs in its own shell and the recipe stops at the first failure.

# List the recipes.
default:
    @just --list

# Create or update .venv with the runtime and dev dependencies.
setup:
    uv sync --locked

# The gate: everything CI checks that does not need a wheel build.
check:
    uv run pre-commit run --all-files
    uv run pytest -m "not live" -q

# Apply ruff autofixes and formatting (what the pre-commit hooks would do).
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Run the tests that do not call a real model provider.
test:
    uv run pytest -m "not live" -q

# Run one test file, e.g. `just test-file tests/test_state.py`.
test-file FILE:
    uv run pytest "{{FILE}}" -q

# Offline smoke run of a persona (bundled id or path to a YAML file).
smoke PERSONA="anxious-newcomer":
    uv run moodsona smoke "{{PERSONA}}"

# Validate the bundled personas against personas/schema.json.
validate:
    uv run moodsona validate
