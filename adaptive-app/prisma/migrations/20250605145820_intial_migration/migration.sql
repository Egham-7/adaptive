BEGIN TRY

BEGIN TRAN;

-- CreateTable
CREATE TABLE [dbo].[Conversation] (
    [id] INT NOT NULL IDENTITY(1,1),
    [title] NVARCHAR(1000) NOT NULL,
    [createdAt] DATETIME2 NOT NULL CONSTRAINT [Conversation_createdAt_df] DEFAULT CURRENT_TIMESTAMP,
    [updatedAt] DATETIME2 NOT NULL,
    [deletedAt] DATETIME2,
    [pinned] BIT NOT NULL CONSTRAINT [Conversation_pinned_df] DEFAULT 0,
    CONSTRAINT [Conversation_pkey] PRIMARY KEY CLUSTERED ([id])
);

-- CreateTable
CREATE TABLE [dbo].[Message] (
    [id] NVARCHAR(1000) NOT NULL,
    [role] NVARCHAR(1000) NOT NULL,
    [content] NVARCHAR(1000) NOT NULL,
    [reasoning] NVARCHAR(1000),
    [data] NVARCHAR(1000),
    [annotations] NVARCHAR(1000),
    [toolInvocations] NVARCHAR(1000),
    [parts] NVARCHAR(1000),
    [experimentalAttachments] NVARCHAR(1000),
    [createdAt] DATETIME2 NOT NULL CONSTRAINT [Message_createdAt_df] DEFAULT CURRENT_TIMESTAMP,
    [updatedAt] DATETIME2 NOT NULL,
    [deletedAt] DATETIME2,
    [conversationId] INT NOT NULL,
    CONSTRAINT [Message_pkey] PRIMARY KEY CLUSTERED ([id])
);

-- CreateIndex
CREATE NONCLUSTERED INDEX [Conversation_deletedAt_idx] ON [dbo].[Conversation]([deletedAt]);

-- CreateIndex
CREATE NONCLUSTERED INDEX [Message_deletedAt_idx] ON [dbo].[Message]([deletedAt]);

-- CreateIndex
CREATE NONCLUSTERED INDEX [Message_conversationId_idx] ON [dbo].[Message]([conversationId]);

-- CreateIndex
CREATE NONCLUSTERED INDEX [Message_role_idx] ON [dbo].[Message]([role]);

-- AddForeignKey
ALTER TABLE [dbo].[Message] ADD CONSTRAINT [Message_conversationId_fkey] FOREIGN KEY ([conversationId]) REFERENCES [dbo].[Conversation]([id]) ON DELETE CASCADE ON UPDATE CASCADE;

COMMIT TRAN;

END TRY
BEGIN CATCH

IF @@TRANCOUNT > 0
BEGIN
    ROLLBACK TRAN;
END;
THROW

END CATCH
