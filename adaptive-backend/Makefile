PROJECT_NAME := adaptive-backend
CMD_DIR := cmd/api
BUILD_DIR := build
ONNX_RUNTIME_VERSION := 1.22.0
ONNX_RUNTIME_ARCH := x64
ONNX_RUNTIME_OS := linux
ONNX_RUNTIME_TAR := onnxruntime-$(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)-$(ONNX_RUNTIME_VERSION).tgz
ONNX_RUNTIME_URL := https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_RUNTIME_VERSION)/$(ONNX_RUNTIME_TAR)
ONNX_RUNTIME_PATH := $(BUILD_DIR)/onnxruntime
ONNX_RUNTIME_LIB_DIR := $(ONNX_RUNTIME_PATH)/lib


.PHONY: all build run clean install-onnx-runtime setup test lint format docker

all: build

build: $(BUILD_DIR)/$(PROJECT_NAME)

$(BUILD_DIR)/$(PROJECT_NAME): $(ONNX_RUNTIME_LIB_DIR)/onnxruntime.so
	@echo "Building $(PROJECT_NAME)..."
	mkdir -p $(BUILD_DIR)/lib # Ensure the lib directory exists for copying .so files
	CGO_LDFLAGS="-Wl,-rpath,'$$ORIGIN/lib'" go build -o $@ $(CMD_DIR)/main.go
	# Copy ONNX Runtime shared libraries next to the executable's 'lib' subdirectory
	cp -L $(ONNX_RUNTIME_LIB_DIR)/*.so* $(BUILD_DIR)/lib/

# Target to prepare the ONNX Runtime library
$(ONNX_RUNTIME_LIB_DIR)/onnxruntime.so:
	@echo "Checking ONNX Runtime installation..."
	@mkdir -p $(ONNX_RUNTIME_PATH)
	# Multi-line shell command block for downloading ONNX Runtime
	# Each internal line (except the very last of the logical block) must end with '\'.
	# The 'fi;' ensures the if block is terminated before the next logical command.
	@if [ ! -f "$(ONNX_RUNTIME_PATH)/$(ONNX_RUNTIME_TAR)" ]; then \
		echo "Downloading ONNX Runtime $(ONNX_RUNTIME_VERSION) for $(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)..."; \
		wget --show-progress -O "$(ONNX_RUNTIME_PATH)/$(ONNX_RUNTIME_TAR)" "$(ONNX_RUNTIME_URL)"; \
	else \
		echo "ONNX Runtime tarball already exists."; \
	fi;
	@echo "Extracting ONNX Runtime..."
	@tar -xzf "$(ONNX_RUNTIME_PATH)/$(ONNX_RUNTIME_TAR)" -C $(ONNX_RUNTIME_PATH) --strip-components=1
	@echo "ONNX Runtime libraries ready at $(ONNX_RUNTIME_LIB_DIR)"

run: $(BUILD_DIR)/$(PROJECT_NAME)
	@echo "Running $(PROJECT_NAME)..."
	# LD_LIBRARY_PATH is generally not needed if -rpath is used correctly,
	# but can be a fallback for some systems or debugging.
	# The '$$ORIGIN' rpath makes the executable look for 'lib' relative to itself.
	$(BUILD_DIR)/$(PROJECT_NAME)

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
