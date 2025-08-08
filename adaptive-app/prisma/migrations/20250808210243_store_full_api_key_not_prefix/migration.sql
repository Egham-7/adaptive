/*
  Warnings:

  - You are about to drop the column `apiKeyPrefix` on the `Provider` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "public"."Provider" DROP COLUMN "apiKeyPrefix",
ADD COLUMN     "apiKey" TEXT;
