FROM golang:1.24-alpine AS builder

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build with CGO disabled
RUN CGO_ENABLED=0 GOOS=linux go build -o adaptive-backend ./cmd/api

# Runtime stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from the builder stage
COPY --from=builder /app/adaptive-backend .
COPY --from=builder /app/internal/config .

# Expose the application port
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["./adaptive-backend"]
