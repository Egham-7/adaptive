"use client";

import { AlertTriangle, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";

export default function UnauthorizedPage() {
	const searchParams = useSearchParams();
	const returnUrl = searchParams.get("return_url") || "/";

	return (
		<div className="flex min-h-screen items-center justify-center bg-background">
			<div className="mx-auto max-w-md px-6 text-center">
				{/* Icon */}
				<div className="mb-6 flex justify-center">
					<div className="rounded-full bg-destructive/10 p-3">
						<AlertTriangle className="h-8 w-8 text-destructive" />
					</div>
				</div>

				{/* Heading */}
				<h1 className="mb-4 font-bold text-2xl text-foreground">
					Access Denied
				</h1>

				{/* Description */}
				<p className="mb-8 text-muted-foreground">
					You need to sign in to access this page. Please sign in with your
					account to continue.
				</p>

				{/* Action buttons */}
				<div className="space-y-4">
					<Link href={`/sign-in?redirect_url=${encodeURIComponent(returnUrl)}`}>
						<Button className="w-full" size="lg">
							Sign In to Continue
						</Button>
					</Link>

					<Link href={`/sign-up?redirect_url=${encodeURIComponent(returnUrl)}`}>
						<Button variant="outline" className="w-full" size="lg">
							Create New Account
						</Button>
					</Link>

					<div className="pt-4">
						<Link href="/">
							<Button variant="ghost" className="flex items-center gap-2">
								<ArrowLeft className="h-4 w-4" />
								Back to Home
							</Button>
						</Link>
					</div>
				</div>

				{/* Help text */}
				<div className="mt-8 text-muted-foreground text-sm">
					<p>
						Need help?{" "}
						<Link
							href="/support"
							className="text-primary underline-offset-4 hover:underline"
						>
							Contact Support
						</Link>
					</p>
				</div>
			</div>
		</div>
	);
}
