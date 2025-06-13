# --- Configuration ---
PROJECT_NAME := adaptive-backend
CMD_DIR := cmd/api
BUILD_DIR := build
ONNX_RUNTIME_VERSION := 1.17.1 # IMPORTANT: Check latest stable release on GitHub
ONNX_RUNTIME_ARCH := x64     # Adjust if your system is different (e.g., arm64)
ONNX_RUNTIME_OS := linux     # Adjust for your OS (linux, macos, windows)
ONNX_RUNTIME_TAR := onnxruntime-$(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)-$(ONNX_RUNTIME_VERSION).tgz
ONNX_RUNTIME_URL := https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_RUNTIME_VERSION)/$(ONNX_RUNTIME_TAR)
ONNX_RUNTIME_PATH := $(BUILD_DIR)/onnxruntime
ONNX_RUNTIME_LIB_DIR := $(ONNX_RUNTIME_PATH)/lib

# --- Build Targets ---

.PHONY: all build run clean install-onnx-runtime setup test lint format docker

all: build

build: $(BUILD_DIR)/$(PROJECT_NAME)

$(BUILD_DIR)/$(PROJECT_NAME): $(ONNX_RUNTIME_LIB_DIR)/onnxruntime.so
	@echo "Building $(PROJECT_NAME)..."
	mkdir -p $(BUILD_DIR)
	CGO_LDFLAGS="-Wl,-rpath,./lib" go build -o $@ $(CMD_DIR)/main.go
	# Copy ONNX Runtime shared libraries next to the executable
	cp -L $(ONNX_RUNTIME_LIB_DIR)/*.so* $(BUILD_DIR)/lib/

# Target to prepare the ONNX Runtime library
$(ONNX_RUNTIME_LIB_DIR)/onnxruntime.so:
	@echo "Checking ONNX Runtime installation..."
	@mkdir -p $(ONNX_RUNTIME_PATH)
	@if [ ! -f "$(ONNX_RUNTIME_PATH)/$(ONNX_RUNTIME_TAR)" ]; then \
		echo "Downloading ONNX Runtime $(ONNX_RUNTIME_VERSION) for $(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)..."; \
		wget -q --show-progress -O $(ONNX_RUNTIME_PATH)/$(ONNX_RUNTIME_TAR) $(ONNX_RUNTIME_URL); \
	else \
		echo "ONNX Runtime tarball already exists."; \
	fi
	@echo "Extracting ONNX Runtime..."
	@tar -xzf $(ONNX_RUNTIME_PATH)/$(ONNX_RUNTIME_TAR) -C $(ONNX_RUNTIME_PATH) --strip-components=1
	@echo "ONNX Runtime libraries ready at $(ONNX_RUNTIME_LIB_DIR)"

run: $(BUILD_DIR)/$(PROJECT_NAME)
	@echo "Running $(PROJECT_NAME)..."
	# Set LD_LIBRARY_PATH for runtime discovery of ONNX Runtime shared libs
	# For Linux, -rpath (used in build) typically makes this unnecessary if libs are bundled.
	# But good for explicit testing or if you don't bundle.
	LD_LIBRARY_PATH=$(BUILD_DIR)/lib $(BUILD_DIR)/$(PROJECT_NAME)

# --- Development & Clean-up ---

clean:
	@echo "Cleaning build artifacts and downloaded models..."
	rm -rf $(BUILD_DIR)
	rm -rf models # If you want to redownload models on clean
	go clean

test:
	@echo "Running tests..."
	go test ./...

lint:
	@echo "Running golangci-lint..."
	golangci-lint run ./...

format:
	@echo "Formatting Go code..."
	go fmt ./...

# --- Docker ---
# Assumes you have a Dockerfile in your project root
docker-build:
	@echo "Building Docker image..."
	docker build -t $(PROJECT_NAME):latest .

docker-run: docker-build
	@echo "Running Docker container..."
	docker run -p 8080:8080 --name $(PROJECT_NAME)-container $(PROJECT_NAME):latest

docker-stop:
	@echo "Stopping Docker container..."
	docker stop $(PROJECT_NAME)-container || true
	docker rm $(PROJECT_NAME)-container || true

docker-clean: docker-stop
	@echo "Cleaning Docker image..."
	docker rmi $(PROJECT_NAME):latest || true
