/*
  Warnings:

  - You are about to drop the `UserCredit` table. If the table is not empty, all the data it contains will be lost.
  - Added the required column `organizationId` to the `CreditTransaction` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE "CreditTransaction" DROP CONSTRAINT "CreditTransaction_userId_fkey";

-- AlterTable (Step 1: Add organizationId as nullable)
ALTER TABLE "CreditTransaction" ADD COLUMN     "organizationId" TEXT;

-- Update organizationId for existing records
UPDATE "CreditTransaction" 
SET "organizationId" = (
  SELECT om."organizationId" 
  FROM "OrganizationMember" om 
  WHERE om."userId" = "CreditTransaction"."userId"
  LIMIT 1
);

-- AlterTable (Step 2: Make organizationId NOT NULL after data is populated)
ALTER TABLE "CreditTransaction" ALTER COLUMN "organizationId" SET NOT NULL;

-- Create OrganizationCredit table first
CREATE TABLE "OrganizationCredit" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "balance" DECIMAL(12,6) NOT NULL DEFAULT 0,
    "totalPurchased" DECIMAL(12,6) NOT NULL DEFAULT 0,
    "totalUsed" DECIMAL(12,6) NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "OrganizationCredit_pkey" PRIMARY KEY ("id")
);

-- Create indexes for OrganizationCredit
CREATE UNIQUE INDEX "OrganizationCredit_organizationId_key" ON "OrganizationCredit"("organizationId");
CREATE INDEX "OrganizationCredit_organizationId_idx" ON "OrganizationCredit"("organizationId");

-- Start transaction for data migration
BEGIN;

-- Check for users without organization membership and handle them
DO $$
BEGIN
    -- Check if any UserCredit records exist for users without organization membership
    IF EXISTS (
        SELECT 1 FROM "UserCredit" uc 
        WHERE NOT EXISTS (
            SELECT 1 FROM "OrganizationMember" om 
            WHERE om."userId" = uc."userId"
        )
    ) THEN
        RAISE EXCEPTION 'Migration cannot proceed: UserCredit records exist for users without organization membership. Please ensure all users are members of an organization before running this migration.';
    END IF;
END $$;

-- Migrate UserCredit data to OrganizationCredit
-- Group user credits by organization and aggregate the balances
INSERT INTO "OrganizationCredit" ("id", "organizationId", "balance", "totalPurchased", "totalUsed", "createdAt", "updatedAt")
SELECT 
    gen_random_uuid() as "id",
    om."organizationId",
    COALESCE(SUM(uc."balance"), 0) as "balance",
    COALESCE(SUM(uc."totalPurchased"), 0) as "totalPurchased", 
    COALESCE(SUM(uc."totalUsed"), 0) as "totalUsed",
    MIN(uc."createdAt") as "createdAt",
    NOW() as "updatedAt"
FROM "UserCredit" uc
INNER JOIN "OrganizationMember" om ON om."userId" = uc."userId"
GROUP BY om."organizationId"
ON CONFLICT ("organizationId") DO UPDATE SET
    "balance" = "OrganizationCredit"."balance" + EXCLUDED."balance",
    "totalPurchased" = "OrganizationCredit"."totalPurchased" + EXCLUDED."totalPurchased",
    "totalUsed" = "OrganizationCredit"."totalUsed" + EXCLUDED."totalUsed",
    "updatedAt" = NOW();

-- Commit the transaction
COMMIT;

-- DropTable
DROP TABLE "UserCredit";

-- CreateIndex
CREATE INDEX "CreditTransaction_organizationId_idx" ON "CreditTransaction"("organizationId");

-- AddForeignKey
ALTER TABLE "OrganizationCredit" ADD CONSTRAINT "OrganizationCredit_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CreditTransaction" ADD CONSTRAINT "CreditTransaction_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "OrganizationCredit"("organizationId") ON DELETE CASCADE ON UPDATE CASCADE;
