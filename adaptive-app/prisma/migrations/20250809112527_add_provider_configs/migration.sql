/*
  Warnings:

  - You are about to drop the column `apiKey` on the `Provider` table. All the data in the column will be lost.
  - You are about to drop the column `isActive` on the `ProviderModel` table. All the data in the column will be lost.
  - You are about to drop the column `type` on the `ProviderModel` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "public"."ProviderModel_isActive_idx";

-- DropIndex
DROP INDEX "public"."ProviderModel_type_idx";

-- AlterTable
ALTER TABLE "public"."Provider" DROP COLUMN "apiKey",
ADD COLUMN     "createdBy" TEXT;

-- AlterTable
ALTER TABLE "public"."ProviderModel" DROP COLUMN "isActive",
DROP COLUMN "type";

-- CreateTable
CREATE TABLE "public"."ProviderConfig" (
    "id" TEXT NOT NULL,
    "projectId" TEXT NOT NULL,
    "providerId" TEXT NOT NULL,
    "displayName" TEXT,
    "providerApiKey" TEXT NOT NULL,
    "customHeaders" JSONB,
    "customSettings" JSONB,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ProviderConfig_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ProviderConfig_projectId_idx" ON "public"."ProviderConfig"("projectId");

-- CreateIndex
CREATE INDEX "ProviderConfig_providerId_idx" ON "public"."ProviderConfig"("providerId");

-- CreateIndex
CREATE UNIQUE INDEX "ProviderConfig_projectId_providerId_key" ON "public"."ProviderConfig"("projectId", "providerId");

-- CreateIndex
CREATE INDEX "Provider_createdBy_idx" ON "public"."Provider"("createdBy");

-- AddForeignKey
ALTER TABLE "public"."ProviderConfig" ADD CONSTRAINT "ProviderConfig_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "public"."Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ProviderConfig" ADD CONSTRAINT "ProviderConfig_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "public"."Provider"("id") ON DELETE CASCADE ON UPDATE CASCADE;
