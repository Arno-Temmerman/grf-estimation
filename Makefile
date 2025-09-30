all: init train test

init:
	pip install -r requirements.txt

train:
	python train.py

test:
	python test.py