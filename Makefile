help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "  dep                                to install project dependencies."
	@echo "  install                            to install the packages needed."
	@echo "  install-no-venv                    to install the packages needed without creating a virtual environment."
	@echo "  install-as-library                 to install the packages needed and the project as a Python package."
	@echo "  pre-commit                         to run the pre-commit checks."
	@echo "  test                               to run the unit tests (CPU-only, synthetic data)."


dep:
	pip install -r poetry-requirements.txt

install: dep
	poetry install --no-interaction --no-root

install-no-venv: dep
	poetry config virtualenvs.create false
	make install-as-library

install-as-library: dep
	poetry install --no-interaction

pre-commit:install
	poetry run pre-commit run ${args}

test:
	poetry run pytest tests/ -q
