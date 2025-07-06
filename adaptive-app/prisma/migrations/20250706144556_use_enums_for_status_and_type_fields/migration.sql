/*
  Warnings:

  - The `status` column on the `Project` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - Changed the type of `status` on the `ApiKey` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `provider` on the `ApiUsage` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `requestType` on the `ApiUsage` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `role` on the `OrganizationMember` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `role` on the `ProjectMember` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- CreateEnum
CREATE TYPE "ApiKeyStatus" AS ENUM ('active', 'inactive', 'revoked');

-- CreateEnum
CREATE TYPE "OrganizationMemberRole" AS ENUM ('owner', 'admin', 'member');

-- CreateEnum
CREATE TYPE "ProjectStatus" AS ENUM ('active', 'inactive', 'paused');

-- CreateEnum
CREATE TYPE "ProjectMemberRole" AS ENUM ('owner', 'admin', 'member');

-- CreateEnum
CREATE TYPE "ProviderType" AS ENUM ('openai', 'anthropic', 'google', 'groq', 'deepseek', 'huggingface');

-- CreateEnum
CREATE TYPE "RequestType" AS ENUM ('completion', 'chat', 'embedding', 'image', 'audio');

-- AlterTable
ALTER TABLE "ApiKey" DROP COLUMN "status",
ADD COLUMN     "status" "ApiKeyStatus" NOT NULL;

-- AlterTable
ALTER TABLE "ApiUsage" DROP COLUMN "provider",
ADD COLUMN     "provider" "ProviderType" NOT NULL,
DROP COLUMN "requestType",
ADD COLUMN     "requestType" "RequestType" NOT NULL;

-- AlterTable
ALTER TABLE "OrganizationMember" DROP COLUMN "role",
ADD COLUMN     "role" "OrganizationMemberRole" NOT NULL;

-- AlterTable
ALTER TABLE "Project" DROP COLUMN "status",
ADD COLUMN     "status" "ProjectStatus" NOT NULL DEFAULT 'active';

-- AlterTable
ALTER TABLE "ProjectMember" DROP COLUMN "role",
ADD COLUMN     "role" "ProjectMemberRole" NOT NULL;

-- CreateIndex
CREATE INDEX "ApiUsage_provider_idx" ON "ApiUsage"("provider");

-- CreateIndex
CREATE INDEX "ApiUsage_requestType_idx" ON "ApiUsage"("requestType");

-- CreateIndex
CREATE INDEX "Project_status_idx" ON "Project"("status");
