.PHONY: cpp

install:
	@pip install --verbose ./python/

editable:
	@pip install scikit-build-core pyproject_metadata pathspec pybind11 ninja cmake
	@pip install --no-build-isolation -ve ./python/

uninstall:
	@pip -v uninstall stddesc

cpp:
	@cmake -Bbuild cpp/
	@cmake --build build -j$(nproc --all)
