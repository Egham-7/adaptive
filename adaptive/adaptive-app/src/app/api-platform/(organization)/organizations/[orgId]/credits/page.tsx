"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { CreditManagement } from "@/app/_components/api-platform/organizations/credit-management";
import { Button } from "@/components/ui/button";

export default function CreditsPage() {
	const { orgId } = useParams<{
		orgId: string;
	}>();

	return (
		<div className="w-full px-6 py-2">
			{/* Back Navigation */}
			<div className="mb-6">
				<Link href={`/api-platform/organizations/${orgId}`}>
					<Button variant="ghost" size="sm">
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Projects
					</Button>
				</Link>
			</div>

			{/* Header Section */}
			<div className="mb-8">
				<h1 className="font-semibold text-3xl text-foreground">
					Credits & Billing
				</h1>
				<p className="text-muted-foreground">
					Manage your organization's credits and billing settings
				</p>
			</div>

			{/* Credit Management Component */}
			<CreditManagement organizationId={orgId} />
		</div>
	);
}
