package config

import (
	"adaptive-backend/internal/models"
	"fmt"
	"log"

	"gorm.io/driver/sqlserver"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
)

var DB *gorm.DB

// Initialize sets up the database connection and migrations
func Initialize(server, database, user, password string) error {
	// Build connection string with TLS configuration
	dsn := fmt.Sprintf(
		"server=%s;database=%s;user id=%s;password=%s;encrypt=true;trustServerCertificate=true",
		server, database, user, password,
	)

	// Configure GORM with SQL Server
	config := &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
		NamingStrategy: schema.NamingStrategy{
			TablePrefix: "",    // Avoid database prefix in queries
			NoLowerCase: false, // Keep default case handling
		},
	}

	// Establish database connection
	var err error
	DB, err = gorm.Open(sqlserver.Open(dsn), config)
	if err != nil {
		return fmt.Errorf("database connection failed: %v", err)
	}

	// Run schema migrations
	err = DB.AutoMigrate(&models.Conversation{}, &models.DBMessage{}, &models.APIKey{})
	if err != nil {
		return fmt.Errorf("schema migration failed: %v", err)
	}

	log.Println("Database initialized successfully")
	return nil
}
