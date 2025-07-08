/*
  Warnings:

  - The values [google] on the enum `ProviderType` will be removed. If these variants are still used in the database, this will fail.
  - You are about to alter the column `cost` on the `ApiUsage` table. The data in that column could be lost. The data in that column will be cast from `DoublePrecision` to `Decimal(12,6)`.
  - You are about to alter the column `inputTokenCost` on the `ProviderModel` table. The data in that column could be lost. The data in that column will be cast from `DoublePrecision` to `Decimal(12,6)`.
  - You are about to alter the column `outputTokenCost` on the `ProviderModel` table. The data in that column could be lost. The data in that column will be cast from `DoublePrecision` to `Decimal(12,6)`.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "ProviderType_new" AS ENUM ('openai', 'anthropic', 'gemini', 'groq', 'grok', 'deepseek', 'huggingface');
ALTER TABLE "ApiUsage" ALTER COLUMN "provider" TYPE "ProviderType_new" USING ("provider"::text::"ProviderType_new");
ALTER TYPE "ProviderType" RENAME TO "ProviderType_old";
ALTER TYPE "ProviderType_new" RENAME TO "ProviderType";
DROP TYPE "ProviderType_old";
COMMIT;

-- AlterTable
ALTER TABLE "ApiUsage" ALTER COLUMN "cost" SET DATA TYPE DECIMAL(12,6);

-- AlterTable
ALTER TABLE "ProviderModel" ALTER COLUMN "inputTokenCost" SET DATA TYPE DECIMAL(12,6),
ALTER COLUMN "outputTokenCost" SET DATA TYPE DECIMAL(12,6);

-- CreateTable
CREATE TABLE "ApiKeyRevealToken" (
    "id" TEXT NOT NULL,
    "apiKeyId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "fullKey" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "revealed" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ApiKeyRevealToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ApiKeyRevealToken_token_key" ON "ApiKeyRevealToken"("token");

-- CreateIndex
CREATE INDEX "ApiKeyRevealToken_token_idx" ON "ApiKeyRevealToken"("token");

-- CreateIndex
CREATE INDEX "ApiKeyRevealToken_expiresAt_idx" ON "ApiKeyRevealToken"("expiresAt");

-- CreateIndex
CREATE INDEX "ApiKeyRevealToken_userId_idx" ON "ApiKeyRevealToken"("userId");

-- AddForeignKey
ALTER TABLE "ApiKeyRevealToken" ADD CONSTRAINT "ApiKeyRevealToken_apiKeyId_fkey" FOREIGN KEY ("apiKeyId") REFERENCES "ApiKey"("id") ON DELETE CASCADE ON UPDATE CASCADE;
