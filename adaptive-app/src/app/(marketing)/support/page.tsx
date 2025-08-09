import type { Metadata } from "next";
import SupportForm from "./support-form";

export const metadata: Metadata = {
	title: "Support | Adaptive - Get Help & Submit Tickets",
	description:
		"Get help with Adaptive AI platform. Submit support tickets, browse FAQ, and contact our technical support team for assistance.",
	keywords: [
		"support",
		"help",
		"technical support",
		"contact",
		"FAQ",
		"troubleshooting",
	],
	openGraph: {
		title: "Support - Get Help with Adaptive AI",
		description:
			"Submit support tickets and get help with the Adaptive AI platform.",
		type: "website",
	},
};

export default function SupportPage() {
	return (
		<main>
			<article className="container mx-auto px-6 py-16">
				<header className="mb-12 text-center">
					<h1 className="mb-4 font-display font-semibold text-4xl lg:text-5xl">
						Support Center
					</h1>
					<p className="mx-auto max-w-2xl text-lg text-muted-foreground">
						Need help? Submit a support ticket and our team will get back to you
						within 24 hours.
					</p>
				</header>

				<div className="mx-auto grid max-w-6xl gap-12 lg:grid-cols-2">
					{/* Support Form */}
					<section>
						<h2 className="mb-6 font-semibold text-2xl">
							Submit a Support Ticket
						</h2>
						<SupportForm />
					</section>

					{/* FAQ Section */}
					<section>
						<h2 className="mb-6 font-semibold text-2xl">
							Frequently Asked Questions
						</h2>
						<div className="space-y-6">
							<details className="group rounded-lg border p-4">
								<summary className="flex cursor-pointer list-none items-center justify-between font-medium">
									How do I get started with the API?
									<span className="text-muted-foreground transition-transform group-open:rotate-180">
										▼
									</span>
								</summary>
								<div className="mt-4 text-muted-foreground">
									<p>
										Sign up for an account, create an organization and project,
										then generate an API key. Use our OpenAI-compatible endpoint
										at{" "}
										<code className="rounded bg-muted px-1">
											https://api.llmadaptive.uk/v1/chat/completions
										</code>
									</p>
								</div>
							</details>

							<details className="group rounded-lg border p-4">
								<summary className="flex cursor-pointer list-none items-center justify-between font-medium">
									What AI providers do you support?
									<span className="text-muted-foreground transition-transform group-open:rotate-180">
										▼
									</span>
								</summary>
								<div className="mt-4 text-muted-foreground">
									<p>
										We support OpenAI, Anthropic, Google AI, Groq, DeepSeek, and
										more. Our intelligent routing automatically selects the best
										provider for your request.
									</p>
								</div>
							</details>

							<details className="group rounded-lg border p-4">
								<summary className="flex cursor-pointer list-none items-center justify-between font-medium">
									How does pricing work?
									<span className="text-muted-foreground transition-transform group-open:rotate-180">
										▼
									</span>
								</summary>
								<div className="mt-4 text-muted-foreground">
									<p>
										We charge the original model cost plus a small markup ($0.10
										per million tokens for Developer plan). Team and Enterprise
										plans have monthly fees plus usage costs.
									</p>
								</div>
							</details>

							<details className="group rounded-lg border p-4">
								<summary className="flex cursor-pointer list-none items-center justify-between font-medium">
									How can I monitor my usage?
									<span className="text-muted-foreground transition-transform group-open:rotate-180">
										▼
									</span>
								</summary>
								<div className="mt-4 text-muted-foreground">
									<p>
										Use our dashboard to track usage, costs, and performance
										metrics in real-time. You can also view detailed analytics
										and cost breakdowns by provider.
									</p>
								</div>
							</details>
						</div>

						<div className="mt-8 rounded-lg bg-muted/50 p-6">
							<h3 className="mb-2 font-medium text-lg">Need More Help?</h3>
							<p className="mb-4 text-muted-foreground">
								Check out our comprehensive documentation or join our community.
							</p>
							<div className="flex gap-4">
								<a
									href="https://docs.llmadaptive.uk"
									target="_blank"
									rel="noopener noreferrer"
									className="text-primary hover:underline"
								>
									Documentation →
								</a>
								<a
									href="https://github.com/Egham-7/adaptive"
									target="_blank"
									rel="noopener noreferrer"
									className="text-primary hover:underline"
								>
									GitHub →
								</a>
							</div>
						</div>
					</section>
				</div>
			</article>
		</main>
	);
}
