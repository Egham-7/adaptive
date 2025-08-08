/*
  Warnings:

  - The `provider` column on the `ApiUsage` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - Changed the type of `provider` on the `ClusterModel` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- AlterTable
ALTER TABLE "public"."ApiUsage" DROP COLUMN "provider",
ADD COLUMN     "provider" TEXT;

-- AlterTable
ALTER TABLE "public"."ClusterModel" DROP COLUMN "provider",
ADD COLUMN     "provider" TEXT NOT NULL;

-- DropEnum
DROP TYPE "public"."ProviderType";

-- CreateTable
CREATE TABLE "public"."ModelCapability" (
    "id" TEXT NOT NULL,
    "providerModelId" TEXT NOT NULL,
    "description" TEXT,
    "maxContextTokens" INTEGER,
    "maxOutputTokens" INTEGER,
    "supportsFunctionCalling" BOOLEAN NOT NULL DEFAULT false,
    "languagesSupported" TEXT,
    "modelSizeParams" TEXT,
    "latencyTier" TEXT,
    "taskType" TEXT,
    "complexity" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ModelCapability_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ModelCapability_providerModelId_key" ON "public"."ModelCapability"("providerModelId");

-- CreateIndex
CREATE INDEX "ModelCapability_latencyTier_idx" ON "public"."ModelCapability"("latencyTier");

-- CreateIndex
CREATE INDEX "ModelCapability_taskType_idx" ON "public"."ModelCapability"("taskType");

-- CreateIndex
CREATE INDEX "ModelCapability_complexity_idx" ON "public"."ModelCapability"("complexity");

-- CreateIndex
CREATE INDEX "ApiUsage_provider_idx" ON "public"."ApiUsage"("provider");

-- CreateIndex
CREATE UNIQUE INDEX "ClusterModel_clusterId_provider_modelName_key" ON "public"."ClusterModel"("clusterId", "provider", "modelName");

-- AddForeignKey
ALTER TABLE "public"."ModelCapability" ADD CONSTRAINT "ModelCapability_providerModelId_fkey" FOREIGN KEY ("providerModelId") REFERENCES "public"."ProviderModel"("id") ON DELETE CASCADE ON UPDATE CASCADE;
