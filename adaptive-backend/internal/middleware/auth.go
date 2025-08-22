package middleware

import (
	"adaptive-backend/internal/config"
	"slices"
	"strings"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/golang-jwt/jwt/v5"
)

// JWTClaims represents the JWT payload structure
type JWTClaims struct {
	APIKey         string `json:"apiKey"`
	UserID         string `json:"userId,omitempty"`
	OrganizationID string `json:"organizationId,omitempty"`
	jwt.RegisteredClaims
}

// JWTAuth creates middleware for JWT authentication
func JWTAuth(cfg *config.Config) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Skip auth in development environment
		if !cfg.IsProduction() {
			return c.Next()
		}

		// Get JWT secret from configuration
		jwtSecret := cfg.Server.JWTSecret
		if jwtSecret == "" {
			fiberlog.Error("JWT_SECRET environment variable is required")
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Server configuration error",
				"code":  fiber.StatusInternalServerError,
			})
		}

		// Get JWT token from Authorization header
		authHeader := c.Get("Authorization")
		if authHeader == "" {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Missing Authorization header",
				"code":  fiber.StatusUnauthorized,
			})
		}

		// Check Bearer token format
		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		if tokenString == authHeader {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Invalid Authorization header format. Expected 'Bearer <token>'",
				"code":  fiber.StatusUnauthorized,
			})
		}

		// Parse and validate JWT token
		token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (any, error) {
			// Validate signing method
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fiber.NewError(fiber.StatusUnauthorized, "Invalid signing method")
			}
			return []byte(jwtSecret), nil
		})

		if err != nil {
			fiberlog.Errorf("JWT parsing error: %v", err)
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Invalid JWT token",
				"code":  fiber.StatusUnauthorized,
			})
		}

		// Validate token and extract claims
		if claims, ok := token.Claims.(*JWTClaims); ok && token.Valid {
			// Validate required claims
			if claims.APIKey == "" {
				return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
					"error": "Invalid JWT claims: missing apiKey",
					"code":  fiber.StatusUnauthorized,
				})
			}

			// Validate issuer
			if claims.Issuer != "adaptive-app" {
				return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
					"error": "Invalid JWT issuer",
					"code":  fiber.StatusUnauthorized,
				})
			}

			// Validate audience - check if "adaptive-backend" is in the audience list
			validAudience := slices.Contains(claims.Audience, "adaptive-backend")
			if !validAudience {
				return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
					"error": "Invalid JWT audience",
					"code":  fiber.StatusUnauthorized,
				})
			}

			// Store claims in context for use in handlers
			c.Locals("jwt_claims", claims)
			c.Locals("api_key", claims.APIKey)
			c.Locals("user_id", claims.UserID)
			c.Locals("organization_id", claims.OrganizationID)

			return c.Next()
		}

		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "Invalid JWT token claims",
			"code":  fiber.StatusUnauthorized,
		})
	}
}
