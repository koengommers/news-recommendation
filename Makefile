format-black:
	@poetry run black src/

format-isort:
	@poetry run isort src/

lint-black:
	@poetry run black src/ --check

lint-isort:
	@poetry run isort src/ --check

lint-ruff:
	@poetry run ruff src/

lint-mypy:
	@poetry run mypy src/

format: format-black format-isort

lint: lint-black lint-isort lint-ruff lint-mypy
