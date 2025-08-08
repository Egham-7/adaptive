/*
  Warnings:

  - The `languagesSupported` column on the `ModelCapability` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - A unique constraint covering the columns `[clusterId,priority]` on the table `ClusterModel` will be added. If there are existing duplicate values, this will fail.

*/
-- DropIndex
DROP INDEX "public"."ClusterModel_priority_idx";

-- AlterTable
ALTER TABLE "public"."ModelCapability" DROP COLUMN "languagesSupported",
ADD COLUMN     "languagesSupported" TEXT[];

-- CreateIndex
CREATE INDEX "ClusterModel_clusterId_priority_idx" ON "public"."ClusterModel"("clusterId", "priority");

-- CreateIndex
CREATE UNIQUE INDEX "ClusterModel_clusterId_priority_key" ON "public"."ClusterModel"("clusterId", "priority");
