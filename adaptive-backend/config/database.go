package config

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/metrics"
	"context"
	"database/sql"
	"fmt"
	"log"
	"sync"
	"time"

	"gorm.io/driver/sqlserver"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
)

var (
	DB          *gorm.DB
	dbMetrics   *metrics.DatabaseMetrics
	metricsOnce sync.Once
)

// DatabaseConfig holds database configuration parameters
type DatabaseConfig struct {
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

// DefaultDatabaseConfig returns sensible defaults for database connection pooling
func DefaultDatabaseConfig() *DatabaseConfig {
	return &DatabaseConfig{
		MaxOpenConns:    25,               // Maximum number of open connections
		MaxIdleConns:    10,               // Maximum number of idle connections
		ConnMaxLifetime: 5 * time.Minute,  // Maximum connection lifetime
		ConnMaxIdleTime: 2 * time.Minute,  // Maximum idle time before closing
	}
}

// Initialize sets up the database connection with optimized pooling and migrations
func Initialize(server, database, user, password string) error {
	return InitializeWithConfig(server, database, user, password, DefaultDatabaseConfig())
}

// initializeMetrics initializes database metrics once
func initializeMetrics() {
	metricsOnce.Do(func() {
		dbMetrics = metrics.NewDatabaseMetrics()
		// Start periodic metrics collection
		go startMetricsCollection()
	})
}

// InitializeWithConfig sets up the database connection with custom configuration
func InitializeWithConfig(server, database, user, password string, config *DatabaseConfig) error {
	// Build optimized connection string with performance settings
	dsn := fmt.Sprintf(
		"server=%s;database=%s;user id=%s;password=%s;encrypt=true;trustServerCertificate=true;connection timeout=30;command timeout=30",
		server, database, user, password,
	)

	// Configure GORM with optimized settings
	gormConfig := &gorm.Config{
		Logger: logger.Default.LogMode(logger.Warn), // Reduce logging overhead in production
		NamingStrategy: schema.NamingStrategy{
			TablePrefix: "",
			NoLowerCase: false,
		},
		PrepareStmt:              true,  // Enable prepared statement caching
		DisableForeignKeyConstraintWhenMigrating: false,
		SkipDefaultTransaction:   true,  // Disable auto-transactions for better performance
	}

	// Establish database connection
	var err error
	DB, err = gorm.Open(sqlserver.Open(dsn), gormConfig)
	if err != nil {
		return fmt.Errorf("database connection failed: %v", err)
	}

	// Configure connection pooling for optimal performance
	sqlDB, err := DB.DB()
	if err != nil {
		return fmt.Errorf("failed to get underlying sql.DB: %v", err)
	}

	// Apply connection pool settings
	sqlDB.SetMaxOpenConns(config.MaxOpenConns)
	sqlDB.SetMaxIdleConns(config.MaxIdleConns)
	sqlDB.SetConnMaxLifetime(config.ConnMaxLifetime)
	sqlDB.SetConnMaxIdleTime(config.ConnMaxIdleTime)

	// Initialize metrics
	initializeMetrics()
	
	// Test the connection
	if err := sqlDB.Ping(); err != nil {
		dbMetrics.QueryErrors.WithLabelValues("ping", "connection_failed").Inc()
		return fmt.Errorf("database ping failed: %v", err)
	}

	// Run schema migrations
	if err := runMigrations(DB); err != nil {
		return fmt.Errorf("schema migration failed: %v", err)
	}

	// Create performance-critical indexes
	if err := createIndexes(DB); err != nil {
		return fmt.Errorf("index creation failed: %v", err)
	}

	// Update initial metrics
	updateConnectionPoolMetrics()
	
	log.Printf("Database initialized successfully with pool config: max_open=%d, max_idle=%d, max_lifetime=%v",
		config.MaxOpenConns, config.MaxIdleConns, config.ConnMaxLifetime)
	
	return nil
}

// runMigrations handles database schema migrations
func runMigrations(db *gorm.DB) error {
	return db.AutoMigrate(&models.APIKey{})
}

// createIndexes creates performance-critical database indexes
func createIndexes(db *gorm.DB) error {
	// Index for API key lookups (most frequent query)
	if err := db.Exec(`
		IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_api_keys_key_hash')
		CREATE INDEX idx_api_keys_key_hash ON api_keys (key_hash)
	`).Error; err != nil {
		return fmt.Errorf("failed to create api_keys key_hash index: %v", err)
	}

	// Index for user-based API key queries
	if err := db.Exec(`
		IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_api_keys_user_id')
		CREATE INDEX idx_api_keys_user_id ON api_keys (user_id)
	`).Error; err != nil {
		return fmt.Errorf("failed to create api_keys user_id index: %v", err)
	}

	// Composite index for active API keys lookup
	if err := db.Exec(`
		IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_api_keys_status_expires')
		CREATE INDEX idx_api_keys_status_expires ON api_keys (status, expires_at)
	`).Error; err != nil {
		return fmt.Errorf("failed to create api_keys status_expires index: %v", err)
	}

	log.Println("Database indexes created successfully")
	return nil
}

// GetDBStats returns database connection pool statistics
func GetDBStats() sql.DBStats {
	if DB == nil {
		return sql.DBStats{}
	}
	
	sqlDB, err := DB.DB()
	if err != nil {
		return sql.DBStats{}
	}
	
	return sqlDB.Stats()
}

// HealthCheck performs a database health check
func HealthCheck() error {
	if DB == nil {
		return fmt.Errorf("database not initialized")
	}
	
	sqlDB, err := DB.DB()
	if err != nil {
		return fmt.Errorf("failed to get underlying sql.DB: %v", err)
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	start := time.Now()
	err = sqlDB.PingContext(ctx)
	
	if dbMetrics != nil {
		duration := time.Since(start).Seconds()
		if err != nil {
			dbMetrics.QueryErrors.WithLabelValues("ping", "timeout").Inc()
			dbMetrics.QueryDuration.WithLabelValues("ping", "health_check").Observe(duration)
		} else {
			dbMetrics.QueryDuration.WithLabelValues("ping", "health_check").Observe(duration)
		}
	}
	
	return err
}

// updateConnectionPoolMetrics updates database connection pool metrics
func updateConnectionPoolMetrics() {
	if DB == nil || dbMetrics == nil {
		return
	}
	
	sqlDB, err := DB.DB()
	if err != nil {
		return
	}
	
	stats := sqlDB.Stats()
	dbMetrics.UpdateConnectionPoolStats(
		stats.MaxOpenConnections,
		stats.OpenConnections,
		stats.InUse,
		stats.Idle,
		stats.WaitCount,
		stats.MaxIdleClosed,
		stats.MaxLifetimeClosed,
	)
	
	// Update individual metrics
	dbMetrics.WaitCount.Add(float64(stats.WaitCount))
	dbMetrics.WaitDuration.Observe(stats.WaitDuration.Seconds())
	dbMetrics.MaxIdleClosed.Add(float64(stats.MaxIdleClosed))
	dbMetrics.MaxLifetimeClosed.Add(float64(stats.MaxLifetimeClosed))
}

// startMetricsCollection starts periodic collection of database metrics
func startMetricsCollection() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		updateConnectionPoolMetrics()
	}
}

// RecordQueryMetrics records metrics for a database query
func RecordQueryMetrics(operation, table string, duration time.Duration, err error) {
	if dbMetrics == nil {
		return
	}
	
	dbMetrics.QueryDuration.WithLabelValues(operation, table).Observe(duration.Seconds())
	
	if err != nil {
		errorType := "unknown"
		if err == context.DeadlineExceeded {
			errorType = "timeout"
		} else if err == sql.ErrNoRows {
			errorType = "not_found"
		} else {
			errorType = "query_error"
		}
		dbMetrics.QueryErrors.WithLabelValues(operation, errorType).Inc()
	}
}

// RecordTransactionMetrics records metrics for a database transaction
func RecordTransactionMetrics(operation string, duration time.Duration, err error) {
	if dbMetrics == nil {
		return
	}
	
	dbMetrics.TransactionDuration.WithLabelValues(operation).Observe(duration.Seconds())
	
	if err != nil {
		dbMetrics.QueryErrors.WithLabelValues("transaction", "failed").Inc()
	}
}