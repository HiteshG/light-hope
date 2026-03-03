# Python Starter

This is an optional Python starter for the challenge. Use it, adapt it, or ignore it entirely and build in any language you prefer.

## Structure

```
starter/
├── pyproject.toml                   # Project config (add your dependencies here)
├── src/
│   └── light_agent/
│       ├── __init__.py
│       ├── types.py                 # Shared types (extend or replace)
│       ├── mock_tools.py            # Mock tool executor (simulates Light API)
│       ├── mock_llm.py              # Mock LLM client (scripted responses)
│       └── runner.py                # Entry point — YOUR CODE GOES HERE
└── tests/
    └── test_scenarios.py            # Test stubs — implement assertions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running

```bash
# Run the entry point
python -m light_agent.runner

# Run tests
pytest tests/ -v
```

## What's Provided vs. What You Build

| File | Provided? | Your job |
|------|-----------|----------|
| `types.py` | ✅ Starting point | Extend or replace with your own types |
| `mock_tools.py` | ✅ Complete | Use as-is (or read to understand the tool behavior) |
| `mock_llm.py` | ✅ Complete | Use for testing; also build a real provider adapter |
| `runner.py` | ⬜ Skeleton | **Build your agent runtime here** |
| `test_scenarios.py` | ⬜ Stubs | **Implement the test assertions** |

## Adding Dependencies

Edit `pyproject.toml` to add any packages you need:

```toml
dependencies = [
    "openai",       # If using OpenAI
    "anthropic",    # If using Anthropic
    "pydantic",     # If you prefer Pydantic models
    # ... etc
]
```

Then `pip install -e ".[dev]"` again.

## Using a Different Language?

Grab the JSON files from `data/` and build from scratch:

- `data/tools.json` — tool definitions (JSON Schema format)
- `data/mock_data.json` — invoice and user data
- `data/scenarios.json` — test scenarios and expected behaviors

The mock tool logic in `mock_tools.py` is straightforward to re-implement — it's just filtering and lookups against the JSON data.
