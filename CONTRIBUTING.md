# Contributing to ORC Observatory

Thanks for your interest in contributing! This project is under active development and we welcome pull requests, bug reports, and feature ideas.

## Getting started

1. Fork the repo and clone your fork.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r experiments/requirements-experiments.txt
   pytest tests/ -v  # make sure everything passes
   ```
3. Create a branch for your change: `git checkout -b my-feature`
4. Make your changes, add tests if applicable, and run the test suite.
5. Open a pull request against `master`.

## What we're looking for

- **New search spaces.** The `SearchSpace` protocol (`lib/search_spaces/protocol.py`) makes it straightforward to add new problem types. Implementing a space means defining `neighbors()`, `fitness()`, and `degree`. Permutation flow-shop, graph coloring, and binary constraint satisfaction are all good candidates.
- **More ILS / metaheuristic variants.** The `lib/ils.py` module uses a simple generator interface. Adding a new algorithm is self-contained.
- **Packaging.** We'd like to make the core library pip-installable (`pip install orc-observatory`). Help with `pyproject.toml`, namespace structure, and CI is very welcome.
- **Tests.** Coverage is thin in places — particularly around TSPLIB/QAPLIB spaces and the experiment pipeline.
- **Documentation.** Docstrings, usage examples, and API reference improvements.

## Code style

- Python 3.11+, type hints everywhere.
- NumPy-style docstrings for public functions.
- No linting configuration is enforced yet; just keep it readable.

## Reporting bugs

Open an issue with:
- What you ran (command, config, Python version).
- What you expected.
- What happened instead (full traceback if applicable).

## Questions

Open a discussion or issue — happy to help.
