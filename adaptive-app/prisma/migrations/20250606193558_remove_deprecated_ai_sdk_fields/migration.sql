/*
  Warnings:

  - You are about to drop the column `data` on the `Message` table. All the data in the column will be lost.
  - You are about to drop the column `toolInvocations` on the `Message` table. All the data in the column will be lost.

*/
BEGIN TRY

BEGIN TRAN;

-- AlterTable
ALTER TABLE [dbo].[Message] DROP COLUMN [data],
[toolInvocations];

COMMIT TRAN;

END TRY
BEGIN CATCH

IF @@TRANCOUNT > 0
BEGIN
    ROLLBACK TRAN;
END;
THROW

END CATCH
