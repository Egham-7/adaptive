-- DropForeignKey
ALTER TABLE "public"."ApiUsage" DROP CONSTRAINT "ApiUsage_apiKeyId_fkey";

-- AlterTable
ALTER TABLE "public"."ApiUsage" ALTER COLUMN "apiKeyId" DROP NOT NULL;

-- AddForeignKey
ALTER TABLE "public"."ApiUsage" ADD CONSTRAINT "ApiUsage_apiKeyId_fkey" FOREIGN KEY ("apiKeyId") REFERENCES "public"."ApiKey"("id") ON DELETE SET NULL ON UPDATE CASCADE;
