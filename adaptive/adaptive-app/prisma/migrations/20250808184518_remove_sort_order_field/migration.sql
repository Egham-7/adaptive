/*
  Warnings:

  - The values [disabled] on the enum `FallbackMode` will be removed. If these variants are still used in the database, this will fail.
  - You are about to drop the column `sortOrder` on the `CreditPackage` table. All the data in the column will be lost.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "public"."FallbackMode_new" AS ENUM ('sequential', 'parallel');
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" DROP DEFAULT;
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" TYPE "public"."FallbackMode_new" USING ("fallbackMode"::text::"public"."FallbackMode_new");
ALTER TYPE "public"."FallbackMode" RENAME TO "FallbackMode_old";
ALTER TYPE "public"."FallbackMode_new" RENAME TO "FallbackMode";
DROP TYPE "public"."FallbackMode_old";
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" SET DEFAULT 'parallel';
COMMIT;

-- DropIndex
DROP INDEX "public"."CreditPackage_sortOrder_idx";

-- AlterTable
ALTER TABLE "public"."CreditPackage" DROP COLUMN "sortOrder";
