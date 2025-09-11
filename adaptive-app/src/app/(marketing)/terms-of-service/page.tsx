import type { Metadata } from "next";

export const metadata: Metadata = {
	title: "Terms of Service | Adaptive",
	description: "Terms of Service for Adaptive LLM infrastructure platform",
};

export default function TermsOfServicePage() {
	return (
		<div className="container mx-auto max-w-4xl px-4 py-12">
			<h1 className="mb-8 font-bold text-4xl">Terms of Service</h1>
			<p className="mb-8 text-muted-foreground">
				Last updated: {new Date().toLocaleDateString()}
			</p>

			<div className="prose prose-gray dark:prose-invert max-w-none space-y-8">
				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						1. Acceptance of Terms
					</h2>
					<p>
						By accessing or using Adaptive's services ("Platform"), you agree to
						be bound by these Terms of Service ("Terms"). If you disagree with
						any part of these terms, you may not access the service.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						2. Description of Service
					</h2>
					<p>
						Adaptive provides an intelligent LLM infrastructure platform that
						routes requests to multiple AI providers including OpenAI,
						Anthropic, Google AI, Meta, Groq, DeepSeek, Cohere, Mistral,
						HuggingFace, and others ("Third-Party Providers"). Our service
						includes:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Multi-provider LLM routing and load balancing</li>
						<li>Chat interface and conversation management</li>
						<li>API key management and authentication</li>
						<li>Usage analytics and cost optimization</li>
						<li>Credit management and billing</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						3. Third-Party Provider Terms
					</h2>
					<div className="mb-4 rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20">
						<p className="font-semibold text-yellow-800 dark:text-yellow-200">
							Important Notice:
						</p>
						<p className="mt-1 text-yellow-700 dark:text-yellow-300">
							When you use our service, your data and requests are processed by
							third-party AI providers. Each provider has their own terms of
							service and data handling practices.
						</p>
					</div>
					<p>By using Adaptive, you acknowledge and agree that:</p>
					<ul className="mt-2 list-disc space-y-2 pl-6">
						<li>
							Your prompts, messages, and generated responses may be processed
							and temporarily stored by Third-Party Providers
						</li>
						<li>
							Each Third-Party Provider may have different data retention and
							privacy policies
						</li>
						<li>
							You are responsible for reviewing and complying with each
							provider's terms of service
						</li>
						<li>
							Adaptive acts as an intermediary and is not responsible for
							Third-Party Provider policies or actions
						</li>
						<li>
							Some providers may use your data to improve their models unless
							you opt out through their respective platforms
						</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						4. User Responsibilities
					</h2>
					<p>You agree to:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Provide accurate and complete information</li>
						<li>Keep your API keys and login credentials secure</li>
						<li>Not share your account with unauthorized parties</li>
						<li>Use the service in compliance with all applicable laws</li>
						<li>Not attempt to circumvent usage limits or security measures</li>
						<li>
							Not use the service for illegal, harmful, or prohibited purposes
						</li>
						<li>Respect Third-Party Provider usage policies and rate limits</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">5. Prohibited Uses</h2>
					<p>You may not use our service to:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Generate illegal, harmful, or abusive content</li>
						<li>Violate any Third-Party Provider's acceptable use policies</li>
						<li>Infringe on intellectual property rights</li>
						<li>Distribute malware or engage in malicious activities</li>
						<li>Impersonate others or provide false information</li>
						<li>Attempt to reverse engineer or manipulate our systems</li>
						<li>Exceed rate limits or abuse the service</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						6. Data Processing and Storage
					</h2>
					<p>We process and store the following types of data:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>
							<strong>Chat Data:</strong> Your conversations, prompts, and AI
							responses are stored to provide the service
						</li>
						<li>
							<strong>Usage Analytics:</strong> Request counts, token usage,
							costs, and performance metrics
						</li>
						<li>
							<strong>Account Information:</strong> Profile data, organization
							details, and billing information
						</li>
						<li>
							<strong>API Keys:</strong> Your Third-Party Provider API keys are
							encrypted and stored securely
						</li>
						<li>
							<strong>Technical Data:</strong> Logs, error reports, and system
							performance data
						</li>
					</ul>
					<p className="mt-4">
						This data may be shared with Third-Party Providers as necessary to
						fulfill your requests. See our Privacy Policy for detailed
						information about data handling.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						7. Billing and Credits
					</h2>
					<p>Our service operates on a credit-based system:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>
							Credits are consumed based on token usage and provider costs
						</li>
						<li>
							Billing is processed through Stripe with standard payment terms
						</li>
						<li>Credits are non-refundable once consumed</li>
						<li>Unused credits may expire according to your plan terms</li>
						<li>You are responsible for monitoring your credit balance</li>
						<li>Service may be suspended if credits are insufficient</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						8. Service Availability
					</h2>
					<p>
						We strive to maintain high availability but cannot guarantee
						uninterrupted service. Service availability depends on:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Third-Party Provider uptime and performance</li>
						<li>Network connectivity and infrastructure</li>
						<li>Scheduled maintenance and updates</li>
						<li>Force majeure events beyond our control</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						9. Intellectual Property
					</h2>
					<p>
						The Adaptive platform, including its code, design, and
						documentation, is protected by intellectual property laws. You
						retain ownership of your input data and generated content, subject
						to Third-Party Provider terms.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						10. Limitation of Liability
					</h2>
					<div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
						<p className="font-semibold">
							TO THE MAXIMUM EXTENT PERMITTED BY LAW, ADAPTIVE SHALL NOT BE
							LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR
							PUNITIVE DAMAGES, INCLUDING WITHOUT LIMITATION, LOSS OF PROFITS,
							DATA, USE, OR OTHER INTANGIBLE LOSSES.
						</p>
					</div>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">11. Indemnification</h2>
					<p>
						You agree to indemnify and hold harmless Adaptive from any claims,
						damages, or expenses arising from:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Your use of the service</li>
						<li>Your violation of these terms</li>
						<li>Your violation of Third-Party Provider terms</li>
						<li>Content you submit or generate</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">12. Termination</h2>
					<p>
						We may terminate or suspend your account immediately, without prior
						notice, if you breach these terms. You may terminate your account at
						any time by contacting us. Upon termination:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Your access to the service will cease</li>
						<li>Data may be deleted according to our retention policy</li>
						<li>Unused credits may be forfeited</li>
						<li>Outstanding charges remain due</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">13. Changes to Terms</h2>
					<p>
						We reserve the right to modify these terms at any time. We will
						notify users of material changes via email or platform
						notifications. Continued use constitutes acceptance of updated
						terms.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">14. Governing Law</h2>
					<p>
						These terms are governed by and construed in accordance with the
						laws of the jurisdiction where Adaptive is incorporated, without
						regard to conflict of law principles.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						15. Contact Information
					</h2>
					<p>
						For questions about these Terms of Service, please contact us
						through our support channels or at the contact information provided
						on our platform.
					</p>
				</section>
			</div>
		</div>
	);
}
