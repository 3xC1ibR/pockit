IMAGE_NAME=pockit
CW_DIR=$(shell pwd)

pipenv-run:
	pipenv run python run.py
