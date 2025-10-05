# Dependency Management

Model Tea uses `pyproject.toml` as the single source of truth for dependencies.

## Installing Dependencies

### Standard Installation

```bash
pip install -e .
```

This installs all runtime dependencies from `pyproject.toml`.

### Development Installation

```bash
pip install -e ".[dev]"
```

This installs runtime + development dependencies (pytest, black, isort, mypy).

## Managing Dependencies

### Adding a Dependency

Edit `pyproject.toml`:

```toml
[project]
dependencies = [
    "torch>=2.0.0,<3.0.0",
    "new-package>=1.0.0",  # Add here
]
```

### Adding a Dev Dependency

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "new-dev-tool>=1.0.0",  # Add here
]
```

### Generating requirements.txt (Optional)

If you need a `requirements.txt` for compatibility:

```bash
pip install pip-tools
pip-compile pyproject.toml
```

**Note:** `pyproject.toml` is the authoritative source. Any generated `requirements.txt` is ignored by git.

## Current Dependencies

### Runtime Dependencies

- **torch** `>=2.0.0,<3.0.0` - PyTorch for model training
- **transformers** `>=4.30.0,<5.0.0` - Hugging Face transformers
- **datasets** `>=2.10.0,<3.0.0` - Dataset handling
- **numpy** `>=1.21.0,<2.0.0` - Numerical operations
- **peft** `>=0.5.0` - Parameter-Efficient Fine-Tuning (LoRA)
- **fastapi** `>=0.104.0` - REST API framework
- **uvicorn** `>=0.24.0` - ASGI server
- **click** `>=8.1.0` - CLI framework
- **rich** `>=13.0.0` - Terminal formatting
- **pydantic** `>=2.0.0` - Data validation
- **python-dotenv** `>=1.0.0` - Environment variables
- **pydantic-settings** `>=2.0.0` - Settings management

### Development Dependencies

- **pytest** `>=7.0.0` - Testing framework
- **pytest-cov** `>=4.0.0` - Coverage reporting
- **black** `>=22.0.0` - Code formatting
- **isort** `>=5.10.0` - Import sorting
- **mypy** `>=1.0.0` - Type checking

## Version Constraints

All dependencies use conservative version constraints:
- Lower bound ensures minimum required features
- Upper bound (major version) prevents breaking changes

Example: `torch>=2.0.0,<3.0.0`
- Minimum: 2.0.0 (features required)
- Maximum: <3.0.0 (avoid breaking changes)

## Updating Dependencies

### Check for Updates

```bash
pip list --outdated
```

### Update a Dependency

1. Test with new version locally
2. Update version constraint in `pyproject.toml`
3. Re-install: `pip install -e .`
4. Run tests: `pytest tests/`
5. Commit changes

### Security Updates

For security vulnerabilities:

```bash
pip install --upgrade package-name
```

Then update the minimum version in `pyproject.toml`.
