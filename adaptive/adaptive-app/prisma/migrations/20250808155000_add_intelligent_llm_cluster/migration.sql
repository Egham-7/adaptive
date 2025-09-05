-- CreateEnum
CREATE TYPE "public"."FallbackMode" AS ENUM ('sequential', 'parallel', 'disabled');

-- AlterTable
ALTER TABLE "public"."ApiUsage" ADD COLUMN     "clusterId" TEXT;

-- CreateTable
CREATE TABLE "public"."LLMCluster" (
    "id" TEXT NOT NULL,
    "projectId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "isDefault" BOOLEAN NOT NULL DEFAULT false,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "fallbackMode" "public"."FallbackMode" NOT NULL DEFAULT 'parallel',
    "enableCircuitBreaker" BOOLEAN NOT NULL DEFAULT true,
    "maxRetries" INTEGER NOT NULL DEFAULT 3,
    "timeoutMs" INTEGER NOT NULL DEFAULT 30000,
    "enableSemanticCache" BOOLEAN NOT NULL DEFAULT true,
    "prioritizeLatency" BOOLEAN NOT NULL DEFAULT false,
    "prioritizeCost" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LLMCluster_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."ClusterModel" (
    "id" TEXT NOT NULL,
    "clusterId" TEXT NOT NULL,
    "provider" "public"."ProviderType" NOT NULL,
    "modelName" TEXT NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 1,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "maxTokens" INTEGER,
    "temperature" DOUBLE PRECISION,
    "topP" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ClusterModel_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LLMCluster_projectId_idx" ON "public"."LLMCluster"("projectId");

-- CreateIndex
CREATE INDEX "LLMCluster_isDefault_idx" ON "public"."LLMCluster"("isDefault");

-- CreateIndex
CREATE UNIQUE INDEX "LLMCluster_projectId_name_key" ON "public"."LLMCluster"("projectId", "name");

-- CreateIndex
CREATE INDEX "ClusterModel_clusterId_idx" ON "public"."ClusterModel"("clusterId");

-- CreateIndex
CREATE INDEX "ClusterModel_priority_idx" ON "public"."ClusterModel"("priority");

-- CreateIndex
CREATE UNIQUE INDEX "ClusterModel_clusterId_provider_modelName_key" ON "public"."ClusterModel"("clusterId", "provider", "modelName");

-- CreateIndex
CREATE INDEX "ApiUsage_clusterId_idx" ON "public"."ApiUsage"("clusterId");

-- AddForeignKey
ALTER TABLE "public"."ApiUsage" ADD CONSTRAINT "ApiUsage_clusterId_fkey" FOREIGN KEY ("clusterId") REFERENCES "public"."LLMCluster"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."LLMCluster" ADD CONSTRAINT "LLMCluster_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "public"."Project"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."ClusterModel" ADD CONSTRAINT "ClusterModel_clusterId_fkey" FOREIGN KEY ("clusterId") REFERENCES "public"."LLMCluster"("id") ON DELETE CASCADE ON UPDATE CASCADE;
