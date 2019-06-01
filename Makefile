clean:
	rm -rf dist/
	rm -rf build/
build:
	python setup.py sdist bdist_wheel
deploy:
	twine upload dist/*
test:
	python -m pytest