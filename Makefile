.PHONY: cpp

editable:
	pip install --verbose --prefix=$(shell python3 -m site --user-base) --editable .

install:
	@pip install --verbose .

uninstall:
	@pip -v uninstall stdesc

cpp:
	@cmake -Bbuild .
	@cmake --build build -j$(nproc --all)
