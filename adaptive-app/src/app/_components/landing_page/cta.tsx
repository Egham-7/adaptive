"use client";

import { SignedIn, SignedOut, SignUpButton } from "@clerk/nextjs";
import { Rocket } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { useSmartRedirect } from "@/hooks/use-smart-redirect";

export default function CallToAction() {
	const redirectPath = useSmartRedirect();

	return (
		<section className="py-16 md:py-32">
			<div className="mx-auto max-w-5xl px-6">
				<div className="text-center">
					<h2 className="text-balance font-display font-semibold text-4xl lg:text-5xl">
						GPT-6 launches. Your router works. Your costs drop. Day one.
					</h2>
					<p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
						The first router that works with tomorrow's models today - no
						training data, no vendor lock-in, 60-90% savings across any
						provider.
					</p>
					<p className="mt-2 font-medium text-primary text-sm">
						Free trial • $3.14 credit • No credit card required
					</p>
					<div className="mt-12 flex flex-wrap justify-center gap-4">
						<SignedOut>
							<SignUpButton
								mode="modal"
								signInForceRedirectUrl="/api-platform/organizations"
							>
								<Button
									size="lg"
									className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
								>
									<Rocket className="relative mr-2 size-4" />
									<span>Start Saving 60-90%</span>
								</Button>
							</SignUpButton>
						</SignedOut>

						<SignedIn>
							<Button
								size="lg"
								className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
								asChild
							>
								<Link href={redirectPath || "/api-platform/organizations"}>
									<Rocket className="relative mr-2 size-4" />
									View Dashboard
								</Link>
							</Button>
						</SignedIn>

						<Button
							asChild
							size="lg"
							variant="outline"
							className="border-primary text-primary hover:bg-primary/10 dark:hover:bg-primary/20"
						>
							<Link href="/features">
								<span>See Technical Details</span>
							</Link>
						</Button>
					</div>
				</div>
			</div>
		</section>
	);
}
