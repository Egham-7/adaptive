# Conversations Module

This module provides comprehensive conversation and message management for the Adaptive Backend. It handles the storage, retrieval, and organization of chat conversations and their associated messages, enabling users to maintain persistent chat histories and access previous interactions.

## Overview

The conversations module consists of two main services that work together to provide full conversation lifecycle management:

- **ConversationService**: Manages conversation metadata, creation, and retrieval
- **MessageService**: Handles individual messages within conversations

## Architecture

```
conversations/
├── conversation_service.go    # Conversation management service
├── message_service.go        # Message management service
└── README.md                # This documentation
```

## Core Components

### ConversationService

The ConversationService manages conversation-level operations including:

- **Creation**: Initialize new conversations with metadata
- **Retrieval**: Fetch conversations by user, ID, or filters
- **Updates**: Modify conversation titles, settings, and metadata
- **Deletion**: Remove conversations and associated data
- **Pagination**: Efficient handling of large conversation lists

### MessageService

The MessageService handles message-level operations including:

- **Storage**: Persist chat messages with proper formatting
- **Retrieval**: Fetch message history for conversations
- **Threading**: Maintain message order and conversation flow
- **Search**: Find messages by content, timestamp, or metadata
- **Analytics**: Track message statistics and usage patterns

## Data Models

### Conversation

```go
type Conversation struct {
    ID          string    `json:"id" gorm:"primaryKey"`
    UserID      string    `json:"user_id" gorm:"index"`
    Title       string    `json:"title"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
    MessageCount int      `json:"message_count" gorm:"-"`
    LastMessage *Message  `json:"last_message,omitempty" gorm:"-"`
    IsArchived  bool      `json:"is_archived" gorm:"default:false"`
    Tags        []string  `json:"tags" gorm:"serializer:json"`
    Metadata    map[string]interface{} `json:"metadata" gorm:"serializer:json"`
}
```

### Message

```go
type Message struct {
    ID             string    `json:"id" gorm:"primaryKey"`
    ConversationID string    `json:"conversation_id" gorm:"index"`
    Role           string    `json:"role"` // user, assistant, system
    Content        string    `json:"content"`
    CreatedAt      time.Time `json:"created_at"`
    TokenCount     int       `json:"token_count"`
    Model          string    `json:"model,omitempty"`
    Provider       string    `json:"provider,omitempty"`
    FinishReason   string    `json:"finish_reason,omitempty"`
    Metadata       map[string]interface{} `json:"metadata" gorm:"serializer:json"`
    
    // Relationships
    Conversation   Conversation `json:"-" gorm:"foreignKey:ConversationID"`
}
```

## Service Interfaces

### ConversationService Interface

```go
type ConversationServiceInterface interface {
    CreateConversation(userID, title string) (*Conversation, error)
    GetConversation(id string, userID string) (*Conversation, error)
    GetUserConversations(userID string, limit, offset int) ([]*Conversation, error)
    UpdateConversation(id string, userID string, updates map[string]interface{}) error
    DeleteConversation(id string, userID string) error
    ArchiveConversation(id string, userID string) error
    SearchConversations(userID string, query string, limit, offset int) ([]*Conversation, error)
    GetConversationStats(userID string) (*ConversationStats, error)
}
```

### MessageService Interface

```go
type MessageServiceInterface interface {
    CreateMessage(conversationID string, message *Message) error
    GetMessages(conversationID string, limit, offset int) ([]*Message, error)
    GetMessage(id string) (*Message, error)
    UpdateMessage(id string, updates map[string]interface{}) error
    DeleteMessage(id string) error
    SearchMessages(conversationID string, query string, limit, offset int) ([]*Message, error)
    GetMessageStats(conversationID string) (*MessageStats, error)
    BulkCreateMessages(conversationID string, messages []*Message) error
}
```

## Usage Examples

### Creating a New Conversation

```go
package main

import (
    "log"
    "adaptive-backend/internal/services/conversations"
)

func createConversation(convService *conversations.ConversationService) {
    conversation, err := convService.CreateConversation(
        "user_123", 
        "Discussion about AI Ethics",
    )
    if err != nil {
        log.Printf("Error creating conversation: %v", err)
        return
    }
    
    log.Printf("Created conversation: %s", conversation.ID)
}
```

### Adding Messages to a Conversation

```go
func addMessage(msgService *conversations.MessageService, conversationID string) {
    message := &models.Message{
        ConversationID: conversationID,
        Role:          "user",
        Content:       "What are the ethical implications of AI?",
        TokenCount:    12,
    }
    
    err := msgService.CreateMessage(conversationID, message)
    if err != nil {
        log.Printf("Error creating message: %v", err)
        return
    }
    
    log.Printf("Added message to conversation: %s", conversationID)
}
```

### Retrieving Conversation History

```go
func getConversationHistory(
    convService *conversations.ConversationService,
    msgService *conversations.MessageService,
    userID string,
) {
    // Get user's conversations
    conversations, err := convService.GetUserConversations(userID, 10, 0)
    if err != nil {
        log.Printf("Error fetching conversations: %v", err)
        return
    }
    
    for _, conv := range conversations {
        // Get messages for each conversation
        messages, err := msgService.GetMessages(conv.ID, 50, 0)
        if err != nil {
            log.Printf("Error fetching messages for %s: %v", conv.ID, err)
            continue
        }
        
        log.Printf("Conversation: %s (%d messages)", conv.Title, len(messages))
        for _, msg := range messages {
            log.Printf("  %s: %s", msg.Role, msg.Content[:50]+"...")
        }
    }
}
```

### Searching Conversations

```go
func searchConversations(convService *conversations.ConversationService, userID string) {
    results, err := convService.SearchConversations(
        userID,
        "machine learning",
        10, // limit
        0,  // offset
    )
    if err != nil {
        log.Printf("Error searching conversations: %v", err)
        return
    }
    
    log.Printf("Found %d conversations about machine learning", len(results))
    for _, conv := range results {
        log.Printf("- %s (ID: %s)", conv.Title, conv.ID)
    }
}
```

## Database Schema

### Conversations Table

```sql
CREATE TABLE conversations (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE,
    tags JSON,
    metadata JSON,
    
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_updated_at (updated_at),
    INDEX idx_user_archived (user_id, is_archived)
);
```

### Messages Table

```sql
CREATE TABLE messages (
    id VARCHAR(255) PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    role ENUM('user', 'assistant', 'system') NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INT DEFAULT 0,
    model VARCHAR(255),
    provider VARCHAR(255),
    finish_reason VARCHAR(255),
    metadata JSON,
    
    INDEX idx_conversation_id (conversation_id),
    INDEX idx_created_at (created_at),
    INDEX idx_role (role),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

## Error Handling

### Common Error Types

```go
var (
    ErrConversationNotFound = errors.New("conversation not found")
    ErrMessageNotFound     = errors.New("message not found")
    ErrUnauthorized        = errors.New("unauthorized access")
    ErrInvalidInput        = errors.New("invalid input data")
    ErrDatabaseError       = errors.New("database operation failed")
)
```

### Error Handling Patterns

```go
func (s *ConversationService) GetConversation(id string, userID string) (*Conversation, error) {
    var conversation Conversation
    
    result := s.db.Where("id = ? AND user_id = ?", id, userID).First(&conversation)
    if result.Error != nil {
        if errors.Is(result.Error, gorm.ErrRecordNotFound) {
            return nil, ErrConversationNotFound
        }
        return nil, fmt.Errorf("failed to get conversation: %w", result.Error)
    }
    
    return &conversation, nil
}
```

## Performance Considerations

### Indexing Strategy

- **Primary Indexes**: ID fields for fast lookups
- **User Indexes**: Efficient user-specific queries
- **Timestamp Indexes**: Chronological sorting and filtering
- **Composite Indexes**: Multi-column queries (user_id + is_archived)

### Query Optimization

```go
// Efficient pagination with cursor-based approach
func (s *ConversationService) GetUserConversationsCursor(
    userID string, 
    cursor string, 
    limit int,
) ([]*Conversation, string, error) {
    query := s.db.Where("user_id = ?", userID).Order("created_at DESC")
    
    if cursor != "" {
        cursorTime, err := time.Parse(time.RFC3339, cursor)
        if err == nil {
            query = query.Where("created_at < ?", cursorTime)
        }
    }
    
    var conversations []*Conversation
    err := query.Limit(limit + 1).Find(&conversations).Error
    if err != nil {
        return nil, "", err
    }
    
    var nextCursor string
    if len(conversations) > limit {
        nextCursor = conversations[limit].CreatedAt.Format(time.RFC3339)
        conversations = conversations[:limit]
    }
    
    return conversations, nextCursor, nil
}
```

### Caching Strategy

```go
type CachedConversationService struct {
    base  ConversationServiceInterface
    cache map[string]*Conversation
    mutex sync.RWMutex
    ttl   time.Duration
}

func (s *CachedConversationService) GetConversation(id string, userID string) (*Conversation, error) {
    // Check cache first
    s.mutex.RLock()
    if cached, exists := s.cache[id]; exists {
        s.mutex.RUnlock()
        return cached, nil
    }
    s.mutex.RUnlock()
    
    // Fetch from database
    conversation, err := s.base.GetConversation(id, userID)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    s.mutex.Lock()
    s.cache[id] = conversation
    s.mutex.Unlock()
    
    return conversation, nil
}
```

## Testing

### Unit Tests

```go
func TestConversationService_CreateConversation(t *testing.T) {
    tests := []struct {
        name     string
        userID   string
        title    string
        wantErr  bool
    }{
        {
            name:    "valid conversation",
            userID:  "user_123",
            title:   "Test Conversation",
            wantErr: false,
        },
        {
            name:    "empty title",
            userID:  "user_123",
            title:   "",
            wantErr: true,
        },
        {
            name:    "empty user ID",
            userID:  "",
            title:   "Test",
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            service := setupTestService(t)
            
            conversation, err := service.CreateConversation(tt.userID, tt.title)
            
            if tt.wantErr {
                assert.Error(t, err)
                assert.Nil(t, conversation)
            } else {
                assert.NoError(t, err)
                assert.NotNil(t, conversation)
                assert.Equal(t, tt.title, conversation.Title)
                assert.Equal(t, tt.userID, conversation.UserID)
            }
        })
    }
}
```

### Integration Tests

```go
func TestConversationFlow_Integration(t *testing.T) {
    db := setupTestDB(t)
    convService := conversations.NewConversationService(db)
    msgService := conversations.NewMessageService(db)
    
    // Create conversation
    conv, err := convService.CreateConversation("user_123", "Integration Test")
    require.NoError(t, err)
    
    // Add messages
    messages := []*models.Message{
        {Role: "user", Content: "Hello"},
        {Role: "assistant", Content: "Hi there!"},
        {Role: "user", Content: "How are you?"},
    }
    
    for _, msg := range messages {
        err := msgService.CreateMessage(conv.ID, msg)
        require.NoError(t, err)
    }
    
    // Retrieve and verify
    retrievedMessages, err := msgService.GetMessages(conv.ID, 10, 0)
    require.NoError(t, err)
    assert.Len(t, retrievedMessages, 3)
    
    // Cleanup
    err = convService.DeleteConversation(conv.ID, "user_123")
    require.NoError(t, err)
}
```

## Monitoring and Observability

### Metrics

```go
var (
    conversationsCreated = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "conversations_created_total",
            Help: "Total number of conversations created",
        },
        []string{"user_type"},
    )
    
    messagesCreated = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "messages_created_total",
            Help: "Total number of messages created",
        },
        []string{"role", "provider"},
    )
    
    conversationDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "conversation_duration_seconds",
            Help: "Duration of conversations",
        },
        []string{"completion_reason"},
    )
)
```

### Logging

```go
func (s *ConversationService) CreateConversation(userID, title string) (*Conversation, error) {
    logger := s.logger.With(
        "operation", "create_conversation",
        "user_id", userID,
        "title_length", len(title),
    )
    
    logger.Info("Creating new conversation")
    
    conversation, err := s.createConversationInDB(userID, title)
    if err != nil {
        logger.Error("Failed to create conversation", "error", err)
        return nil, err
    }
    
    logger.Info("Conversation created successfully", 
        "conversation_id", conversation.ID,
    )
    
    conversationsCreated.WithLabelValues("standard").Inc()
    
    return conversation, nil
}
```

## Security Considerations

### Access Control

```go
func (s *ConversationService) validateUserAccess(conversationID, userID string) error {
    var count int64
    err := s.db.Model(&Conversation{}).
        Where("id = ? AND user_id = ?", conversationID, userID).
        Count(&count).Error
    
    if err != nil {
        return fmt.Errorf("access validation failed: %w", err)
    }
    
    if count == 0 {
        return ErrUnauthorized
    }
    
    return nil
}
```

### Data Sanitization

```go
func sanitizeInput(input string) string {
    // Remove potentially harmful characters
    input = strings.TrimSpace(input)
    input = html.EscapeString(input)
    
    // Limit length
    if len(input) > 10000 {
        input = input[:10000]
    }
    
    return input
}
```

## Configuration

### Service Configuration

```go
type Config struct {
    DB              *gorm.DB
    Logger          *slog.Logger
    CacheEnabled    bool
    CacheTTL        time.Duration
    MaxMessageSize  int
    MaxConversation int
}

func NewConversationService(config *Config) *ConversationService {
    return &ConversationService{
        db:              config.DB,
        logger:          config.Logger,
        cacheEnabled:    config.CacheEnabled,
        cacheTTL:        config.CacheTTL,
        maxMessageSize:  config.MaxMessageSize,
        maxConversations: config.MaxConversations,
    }
}
```

## Future Enhancements

### Planned Features

- **Real-time Updates**: WebSocket support for live conversation updates
- **Message Threading**: Support for branched conversations
- **Conversation Templates**: Pre-defined conversation starters
- **Advanced Search**: Full-text search with relevance scoring
- **Export/Import**: Conversation backup and restoration
- **Collaboration**: Shared conversations between users
- **Analytics Dashboard**: Conversation insights and statistics

### Performance Improvements

- **Database Sharding**: Horizontal scaling for large datasets
- **Read Replicas**: Separate read/write database instances
- **Message Compression**: Compress large message content
- **Async Processing**: Background tasks for heavy operations
- **CDN Integration**: Cache static conversation data