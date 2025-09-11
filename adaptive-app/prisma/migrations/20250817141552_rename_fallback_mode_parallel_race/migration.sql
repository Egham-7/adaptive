/*
  Warnings:

  - The values [parallel] on the enum `FallbackMode` will be removed. If these variants are still used in the database, this will fail.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "public"."FallbackMode_new" AS ENUM ('sequential', 'race');
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" DROP DEFAULT;
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" TYPE "public"."FallbackMode_new" USING ("fallbackMode"::text::"public"."FallbackMode_new");
ALTER TYPE "public"."FallbackMode" RENAME TO "FallbackMode_old";
ALTER TYPE "public"."FallbackMode_new" RENAME TO "FallbackMode";
DROP TYPE "public"."FallbackMode_old";
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" SET DEFAULT 'race';
COMMIT;

-- AlterTable
ALTER TABLE "public"."LLMCluster" ALTER COLUMN "fallbackMode" SET DEFAULT 'race';
