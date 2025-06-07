BEGIN TRY

BEGIN TRAN;

-- AlterTable
ALTER TABLE [dbo].[Message] ALTER COLUMN [content] NVARCHAR(max) NOT NULL;
ALTER TABLE [dbo].[Message] ALTER COLUMN [reasoning] NVARCHAR(max) NULL;
ALTER TABLE [dbo].[Message] ALTER COLUMN [data] NVARCHAR(max) NULL;
ALTER TABLE [dbo].[Message] ALTER COLUMN [annotations] NVARCHAR(max) NULL;
ALTER TABLE [dbo].[Message] ALTER COLUMN [toolInvocations] NVARCHAR(max) NULL;
ALTER TABLE [dbo].[Message] ALTER COLUMN [parts] NVARCHAR(max) NULL;
ALTER TABLE [dbo].[Message] ALTER COLUMN [experimentalAttachments] NVARCHAR(max) NULL;

COMMIT TRAN;

END TRY
BEGIN CATCH

IF @@TRANCOUNT > 0
BEGIN
    ROLLBACK TRAN;
END;
THROW

END CATCH
