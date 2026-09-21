VENV ?= ./.venv/bin

.PHONY: check test unit identity resume tasks

check:
	$(VENV)/python scripts/check_repository.py

test:
	$(VENV)/python -m pytest tests/ -q

unit:
	$(VENV)/python -m pytest tests/unit -q

identity:
	$(VENV)/python -m pytest tests/identity -q

resume:
	$(VENV)/python -m pytest tests/resume -q

tasks:
	$(VENV)/python -m sciel.cli.main tasks validate
