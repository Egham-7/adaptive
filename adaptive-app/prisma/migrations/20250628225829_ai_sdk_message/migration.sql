/*
  Warnings:

  - You are about to drop the column `content` on the `Message` table. All the data in the column will be lost.
  - You are about to drop the column `experimentalAttachments` on the `Message` table. All the data in the column will be lost.
  - You are about to drop the column `reasoning` on the `Message` table. All the data in the column will be lost.
  - Made the column `parts` on table `Message` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "Message" DROP COLUMN "content",
DROP COLUMN "experimentalAttachments",
DROP COLUMN "reasoning",
ADD COLUMN     "metadata" JSONB,
ALTER COLUMN "parts" SET NOT NULL;
