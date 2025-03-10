package config

import (
	"adaptive-backend/internal/models"
	"log"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var DB *gorm.DB

// Initialize sets up the database connection and migrations
func Initialize(dbPath string) error {
	var err error

	// Open SQLite database
	DB, err = gorm.Open(sqlite.Open(dbPath), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return err
	}

	// Auto migrate the schema
	err = DB.AutoMigrate(&models.Conversation{}, &models.DBMessage{})
	if err != nil {
		return err
	}

	log.Println("Database initialized successfully")
	return nil
}
