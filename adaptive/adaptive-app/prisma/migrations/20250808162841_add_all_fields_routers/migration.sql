/*
  Warnings:

  - You are about to drop the column `isDefault` on the `LLMCluster` table. All the data in the column will be lost.
  - You are about to drop the column `prioritizeCost` on the `LLMCluster` table. All the data in the column will be lost.
  - You are about to drop the column `prioritizeLatency` on the `LLMCluster` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "public"."LLMCluster_isDefault_idx";

-- AlterTable
ALTER TABLE "public"."LLMCluster" DROP COLUMN "isDefault",
DROP COLUMN "prioritizeCost",
DROP COLUMN "prioritizeLatency",
ADD COLUMN     "complexityThreshold" DOUBLE PRECISION,
ADD COLUMN     "costBias" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
ADD COLUMN     "enablePromptCache" BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN     "fallbackEnabled" BOOLEAN NOT NULL DEFAULT true,
ADD COLUMN     "promptCacheTTL" INTEGER NOT NULL DEFAULT 3600,
ADD COLUMN     "semanticThreshold" DOUBLE PRECISION NOT NULL DEFAULT 0.85,
ADD COLUMN     "tokenThreshold" INTEGER;
