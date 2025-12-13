# Contributing to CIGRL

Thank you for your interest in contributing to CIGRL! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check [existing issues](https://github.com/TerexSpace/RL_for_Multi_Drone_UN/issues) to avoid duplicates
2. Open a new issue with a clear title and description
3. Include steps to reproduce (for bugs)
4. Add relevant labels (bug, enhancement, question)

### Submitting Code Changes

1. **Fork the repository** and create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style:

   - Use [Black](https://github.com/psf/black) for Python formatting
   - Add docstrings to all public functions
   - Write tests for new functionality

3. **Run tests** before committing:

   ```bash
   pytest tests/ -v
   ```

4. **Submit a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots for UI changes (if applicable)

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Run `flake8` and `black` before committing

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/RL_for_Multi_Drone_UN.git
cd RL_for_Multi_Drone_UN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Seeking Support

- **Documentation**: Check the [README.md](README.md) first
- **Discussions**: Open a [GitHub Discussion](https://github.com/TerexSpace/RL_for_Multi_Drone_UN/discussions) for questions
- **Email**: Contact maintainers for sensitive issues

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
