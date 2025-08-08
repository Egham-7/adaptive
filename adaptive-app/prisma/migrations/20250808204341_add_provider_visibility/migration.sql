/*
  Warnings:

  - A unique constraint covering the columns `[projectId,name]` on the table `Provider` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateEnum
CREATE TYPE "public"."ProviderVisibility" AS ENUM ('system', 'project', 'organization', 'community');

-- DropIndex
DROP INDEX "public"."Provider_name_key";

-- AlterTable
ALTER TABLE "public"."Provider" ADD COLUMN     "apiKeyPrefix" TEXT,
ADD COLUMN     "authHeaderName" TEXT,
ADD COLUMN     "authType" TEXT,
ADD COLUMN     "baseUrl" TEXT,
ADD COLUMN     "headers" JSONB,
ADD COLUMN     "healthEndpoint" TEXT,
ADD COLUMN     "projectId" TEXT,
ADD COLUMN     "rateLimitRpm" INTEGER,
ADD COLUMN     "retryConfig" JSONB,
ADD COLUMN     "timeoutMs" INTEGER,
ADD COLUMN     "visibility" "public"."ProviderVisibility" NOT NULL DEFAULT 'project';

-- CreateIndex
CREATE INDEX "Provider_projectId_idx" ON "public"."Provider"("projectId");

-- CreateIndex
CREATE INDEX "Provider_visibility_idx" ON "public"."Provider"("visibility");

-- CreateIndex
CREATE UNIQUE INDEX "Provider_projectId_name_key" ON "public"."Provider"("projectId", "name");

-- AddForeignKey
ALTER TABLE "public"."Provider" ADD CONSTRAINT "Provider_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "public"."Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;
