package middleware

import (
	"bytes"
	"net/http"
	"os"

	"github.com/clerk/clerk-sdk-go/v2"
	clerkhttp "github.com/clerk/clerk-sdk-go/v2/http"
	"github.com/gofiber/fiber/v2"
	"github.com/valyala/fasthttp/fasthttpadaptor"
)

// ClerkAuthenticator handles authentication using Clerk's session system
type ClerkAuthenticator struct {
	secretKey string
}

// NewClerkAuthenticator creates an authenticator configured with the provided secret key
func NewClerkAuthenticator(secretKey string) *ClerkAuthenticator {
	return &ClerkAuthenticator{
		secretKey: secretKey,
	}
}

// AuthMiddleware returns a Fiber middleware that authenticates requests using Clerk.
// It verifies the bearer token from the Authorization header and makes the
// authenticated user's ID available in the Fiber context locals as "userID".
func AuthMiddleware() fiber.Handler {
	secretKey := os.Getenv("CLERK_SECRET_KEY")
	auth := NewClerkAuthenticator(secretKey)
	return auth.Middleware()
}

// Middleware returns a Fiber handler function that authenticates requests
func (ca *ClerkAuthenticator) Middleware() fiber.Handler {
	clerk.SetKey(ca.secretKey)

	return func(c *fiber.Ctx) error {
		// Convert Fiber/fasthttp request to http.Request
		httpReq := &http.Request{}
		ctx := c.Context()

		// Use fasthttpadaptor to convert the request
		if err := fasthttpadaptor.ConvertRequest(ctx, httpReq, false); err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to process authentication",
			})
		}

		// Add Authorization header if it exists in the original request
		if authHeader := c.Get("Authorization"); authHeader != "" {
			httpReq.Header.Set("Authorization", authHeader)
		}

		// Create a response recorder to capture Clerk middleware results
		recorder := newResponseRecorder()

		// Create a handler to process Clerk results
		httpHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			claims, ok := clerk.SessionClaimsFromContext(r.Context())
			if !ok {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}

			// Store user ID for downstream handlers
			if claims != nil {
				c.Locals("userID", claims.Subject)
			}
		})

		// Apply Clerk middleware
		clerkHandler := clerkhttp.WithHeaderAuthorization()(httpHandler)
		clerkHandler.ServeHTTP(recorder, httpReq)

		// Check if authentication was successful
		if recorder.statusCode != http.StatusOK {
			return c.Status(recorder.statusCode).JSON(fiber.Map{
				"error": "Invalid session",
			})
		}

		return c.Next()
	}
}

// responseRecorder is a minimal implementation of http.ResponseWriter
// that captures status codes
type responseRecorder struct {
	headers    http.Header
	statusCode int
	body       bytes.Buffer
}

// newResponseRecorder creates a new response recorder
func newResponseRecorder() *responseRecorder {
	return &responseRecorder{
		headers:    make(http.Header),
		statusCode: http.StatusOK,
	}
}

// Header returns the header map for setting HTTP headers
func (r *responseRecorder) Header() http.Header {
	return r.headers
}

// Write captures response body data
func (r *responseRecorder) Write(bytes []byte) (int, error) {
	return r.body.Write(bytes)
}

// WriteHeader captures the status code
func (r *responseRecorder) WriteHeader(statusCode int) {
	r.statusCode = statusCode
}

