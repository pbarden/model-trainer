# Contributing to Model Tea

Thank you for your interest in contributing to Model Tea! This document provides guidelines for contributing to this CPU-optimized novel training system.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git for version control

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/chaiq-llc/model-tea.git
cd model-tea
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e .[dev]
```

4. Run tests to ensure everything works:
```bash
pytest
```

## Development Workflow

### Code Style

We use Black for code formatting and follow PEP 8 guidelines:

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### Testing

- Write tests for all new functionality
- Ensure existing tests pass
- Aim for good test coverage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_utils.py
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom memory types
fix: resolve memory allocation issue in large novels
docs: update configuration documentation
test: add integration tests for training pipeline
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Python version and operating system
- Model Tea version
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and logs

### Feature Requests

For new features, please provide:

- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing functionality

### Code Contributions

1. **Fork the repository** and create a feature branch
2. **Write tests** for your changes
3. **Ensure tests pass** and code follows style guidelines
4. **Update documentation** if needed
5. **Submit a pull request** with clear description

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the version numbers following semantic versioning
3. Ensure the PR description clearly describes the changes
4. Link any relevant issues
5. Ensure all status checks pass

## Code Review Guidelines

### For Contributors

- Keep PRs focused and reasonably sized
- Include tests for new functionality
- Update documentation for API changes
- Be responsive to feedback

### For Reviewers

- Provide constructive feedback
- Focus on code quality, correctness, and maintainability
- Consider performance implications
- Ensure changes align with project goals

## Architecture Guidelines

### CPU Optimization

Model Tea focuses on CPU-optimized training. When contributing:

- Consider memory usage and CPU efficiency
- Avoid GPU-specific optimizations unless configurable
- Profile performance-critical code changes
- Document any performance implications

### Memory System

The episodic memory system is a core feature:

- Understand the memory types and retrieval mechanisms
- Test memory functionality thoroughly
- Consider backward compatibility with existing memories
- Document memory-related changes clearly

### Configuration Management

- Use the unified configuration system
- Avoid hardcoded values
- Provide reasonable defaults
- Document configuration options

## Documentation

### Code Documentation

- Use clear, descriptive docstrings
- Follow Google or NumPy docstring format
- Include parameter types and return values
- Provide usage examples for complex functions

### User Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Keep documentation current with code changes
- Use clear, beginner-friendly language

## Release Process

1. Version bumping follows semantic versioning
2. Release notes summarize changes
3. All tests must pass
4. Documentation must be updated
5. Performance regression testing

## Getting Help

- Create an issue for questions
- Join discussions in existing issues
- Check existing documentation
- Review similar implementations

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## License

By contributing to Model Tea, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation for major features

Thank you for contributing to Model Tea!