# Contributing

Thanks for your interest in improving Retail Analytics Copilot.

## Development setup

1. Fork the repository and create a branch from `master`.
2. Create and activate a Python 3.10+ virtual environment.
3. Install the runtime dependencies with `python -m pip install -r requirements.txt`.
4. Run `python -m unittest discover -s tests -v` before opening a pull request.

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests when behavior changes.
- Explain the problem, the solution, and any trade-offs in the description.
- Do not commit API keys, local databases, generated outputs, or environment files.
- Make sure the CI checks pass.

Bug reports and feature proposals are welcome through GitHub Issues. Include steps to reproduce, expected behavior, actual behavior, and relevant environment details.
