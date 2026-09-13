.PHONY: install test train train-ci demo lint-paths

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

train:
	python -m motioncode.cli train --config configs/synthetic.yaml

train-ci:
	python -m motioncode.cli train --config configs/ci.yaml

demo:
	python -m motioncode.cli demo --config configs/ci.yaml
