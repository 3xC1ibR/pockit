IMAGE_NAME=pockit
CW_DIR=$(shell pwd)

pipenv-run:
	pipenv run python run.py


setup:
	pipenv install && \
	echo 'install tf 1.14.0 manually' && \
	pipenv run pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl
