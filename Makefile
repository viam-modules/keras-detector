setup:
	./setup.sh

build:
	./build.sh

lint:
	pylint --disable=C0301,C0303,R0913,W0201,C0114,C0115 src/