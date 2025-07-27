/*
  Warnings:

  - You are about to drop the `UserCredit` table. If the table is not empty, all the data it contains will be lost.
  - Added the required column `organizationId` to the `CreditTransaction` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE "CreditTransaction" DROP CONSTRAINT "CreditTransaction_userId_fkey";

-- AlterTable
ALTER TABLE "CreditTransaction" ADD COLUMN     "organizationId" TEXT NOT NULL;

-- DropTable
DROP TABLE "UserCredit";

-- CreateTable
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

-- CreateIndex
CREATE UNIQUE INDEX "OrganizationCredit_organizationId_key" ON "OrganizationCredit"("organizationId");

-- CreateIndex
CREATE INDEX "OrganizationCredit_organizationId_idx" ON "OrganizationCredit"("organizationId");

-- CreateIndex
CREATE INDEX "CreditTransaction_organizationId_idx" ON "CreditTransaction"("organizationId");

-- AddForeignKey
ALTER TABLE "OrganizationCredit" ADD CONSTRAINT "OrganizationCredit_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CreditTransaction" ADD CONSTRAINT "CreditTransaction_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "OrganizationCredit"("organizationId") ON DELETE CASCADE ON UPDATE CASCADE;
