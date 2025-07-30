# Check for Windows first using the OS environment variable
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    # For Unix-like systems, use uname -s to get the kernel name
    # and convert it to lowercase for consistency
    DETECTED_OS := $(shell uname -s | tr A-Z a-z)
endif

# Target to echo the detected operating system
echo_os:
	@echo "Detected Operating System: $(DETECTED_OS)"

setup:
	ifeq ($(DETECTED_OS), Windows)
		@echo "Setting up for Windows..."
		./setup_windows.sh
	else 
		./setup.sh
	endif

build:
	ifeq ($(DETECTED_OS), Windows)
		@echo "Building for Windows..."
		./build_windows.sh
	else 
		./build.sh
	endif

lint:
	pylint --disable=C0301,C0303,C0116,C0103,R0913,W0201,C0114,C0115 src/

test:
	PYTHONPATH=./src pytest tests/ 