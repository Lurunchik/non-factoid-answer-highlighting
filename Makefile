CODE = highlighting
VENV = poetry run
WIDTH = 120

.PHONY: pretty lint

pretty:
	$(VENV) black  --skip-string-normalization --line-length $(WIDTH) $(CODE) $(TESTS)
	$(VENV) isort --apply --recursive --line-width $(WIDTH) $(CODE) $(TESTS)
	$(VENV) unify --in-place --recursive $(CODE) $(TESTS)

lint:
	$(VENV) black --check --skip-string-normalization --line-length $(WIDTH) $(CODE) $(TESTS)
	$(VENV) flake8 --statistics --max-line-length $(WIDTH) $(CODE) $(TESTS)
	$(VENV) pylint --rcfile=setup.cfg $(CODE)
	$(VENV) mypy $(CODE)
