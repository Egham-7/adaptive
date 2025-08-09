/*
  Warnings:

  - You are about to drop the column `isActive` on the `CreditPackage` table. All the data in the column will be lost.
  - You are about to drop the column `isActive` on the `LLMCluster` table. All the data in the column will be lost.
  - You are about to drop the column `isActive` on the `Provider` table. All the data in the column will be lost.
  - You are about to drop the column `isActive` on the `ProviderConfig` table. All the data in the column will be lost.
  - You are about to alter the column `providerApiKey` on the `ProviderConfig` table. The data in that column could be lost. The data in that column will be cast from `Text` to `VarChar(500)`.
  - You are about to drop the `ClusterModel` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "public"."ClusterModel" DROP CONSTRAINT "ClusterModel_clusterId_fkey";

-- DropIndex
DROP INDEX "public"."CreditPackage_isActive_idx";

-- DropIndex
DROP INDEX "public"."Provider_isActive_idx";

-- AlterTable
ALTER TABLE "public"."CreditPackage" DROP COLUMN "isActive";

-- AlterTable
ALTER TABLE "public"."LLMCluster" DROP COLUMN "isActive";

-- AlterTable
ALTER TABLE "public"."Provider" DROP COLUMN "isActive";

-- AlterTable
ALTER TABLE "public"."ProviderConfig" DROP COLUMN "isActive",
ALTER COLUMN "providerApiKey" SET DATA TYPE VARCHAR(500);

-- DropTable
DROP TABLE "public"."ClusterModel";

-- CreateTable
CREATE TABLE "public"."ClusterProvider" (
    "id" TEXT NOT NULL,
    "clusterId" TEXT NOT NULL,
    "providerId" TEXT NOT NULL,
    "configId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ClusterProvider_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."ProviderConfigModel" (
    "id" TEXT NOT NULL,
    "providerConfigId" TEXT NOT NULL,
    "providerModelId" TEXT NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ProviderConfigModel_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ClusterProvider_clusterId_idx" ON "public"."ClusterProvider"("clusterId");

-- CreateIndex
CREATE INDEX "ClusterProvider_providerId_idx" ON "public"."ClusterProvider"("providerId");

-- CreateIndex
CREATE INDEX "ClusterProvider_configId_idx" ON "public"."ClusterProvider"("configId");

-- CreateIndex
CREATE UNIQUE INDEX "ClusterProvider_clusterId_providerId_key" ON "public"."ClusterProvider"("clusterId", "providerId");

-- CreateIndex
CREATE INDEX "ProviderConfigModel_providerConfigId_idx" ON "public"."ProviderConfigModel"("providerConfigId");

-- CreateIndex
CREATE INDEX "ProviderConfigModel_providerModelId_idx" ON "public"."ProviderConfigModel"("providerModelId");

-- CreateIndex
CREATE INDEX "ProviderConfigModel_priority_idx" ON "public"."ProviderConfigModel"("priority");

-- CreateIndex
CREATE UNIQUE INDEX "ProviderConfigModel_providerConfigId_providerModelId_key" ON "public"."ProviderConfigModel"("providerConfigId", "providerModelId");

-- CreateIndex
CREATE INDEX "ModelCapability_maxContextTokens_idx" ON "public"."ModelCapability"("maxContextTokens");

-- CreateIndex
CREATE INDEX "ModelCapability_supportsFunctionCalling_idx" ON "public"."ModelCapability"("supportsFunctionCalling");

-- CreateIndex
CREATE INDEX "ProviderModel_inputTokenCost_idx" ON "public"."ProviderModel"("inputTokenCost");

-- CreateIndex
CREATE INDEX "ProviderModel_outputTokenCost_idx" ON "public"."ProviderModel"("outputTokenCost");

-- AddForeignKey
ALTER TABLE "public"."ClusterProvider" ADD CONSTRAINT "ClusterProvider_clusterId_fkey" FOREIGN KEY ("clusterId") REFERENCES "public"."LLMCluster"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ClusterProvider" ADD CONSTRAINT "ClusterProvider_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "public"."Provider"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ClusterProvider" ADD CONSTRAINT "ClusterProvider_configId_fkey" FOREIGN KEY ("configId") REFERENCES "public"."ProviderConfig"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ProviderConfigModel" ADD CONSTRAINT "ProviderConfigModel_providerConfigId_fkey" FOREIGN KEY ("providerConfigId") REFERENCES "public"."ProviderConfig"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ProviderConfigModel" ADD CONSTRAINT "ProviderConfigModel_providerModelId_fkey" FOREIGN KEY ("providerModelId") REFERENCES "public"."ProviderModel"("id") ON DELETE CASCADE ON UPDATE CASCADE;
