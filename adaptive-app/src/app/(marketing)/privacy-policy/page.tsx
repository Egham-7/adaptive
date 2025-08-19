import type { Metadata } from "next";

export const metadata: Metadata = {
	title: "Privacy Policy | Adaptive",
	description: "Privacy Policy for Adaptive LLM infrastructure platform",
};

export default function PrivacyPolicyPage() {
	return (
		<div className="container mx-auto max-w-4xl px-4 py-12">
			<h1 className="mb-8 font-bold text-4xl">Privacy Policy</h1>
			<p className="mb-8 text-muted-foreground">
				Last updated: {new Date().toLocaleDateString()}
			</p>

			<div className="prose prose-gray dark:prose-invert max-w-none space-y-8">
				<section>
					<h2 className="mb-4 font-semibold text-2xl">1. Introduction</h2>
					<p>
						Adaptive ("we," "our," or "us") is committed to protecting your
						privacy. This Privacy Policy explains how we collect, use, disclose,
						and safeguard your information when you use our LLM infrastructure
						platform ("Service"). Please read this privacy policy carefully.
					</p>
					<div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
						<p className="font-semibold text-blue-800 dark:text-blue-200">
							Key Point:
						</p>
						<p className="mt-1 text-blue-700 dark:text-blue-300">
							Our platform routes your requests to third-party AI providers.
							Your data may be processed by these providers according to their
							own privacy policies.
						</p>
					</div>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						2. Information We Collect
					</h2>

					<h3 className="mb-3 font-medium text-xl">
						2.1 Information You Provide
					</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>Account Information:</strong> Name, email address,
							organization details
						</li>
						<li>
							<strong>Authentication Data:</strong> Login credentials, API keys
							for third-party providers
						</li>
						<li>
							<strong>Chat Content:</strong> Your prompts, messages,
							conversations, and AI responses
						</li>
						<li>
							<strong>Payment Information:</strong> Billing details processed
							securely through Stripe
						</li>
						<li>
							<strong>Support Communications:</strong> Messages sent through our
							support channels
						</li>
					</ul>

					<h3 className="mt-6 mb-3 font-medium text-xl">
						2.2 Automatically Collected Information
					</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>Usage Analytics:</strong> API request counts, token usage,
							response times, error rates
						</li>
						<li>
							<strong>Technical Data:</strong> IP addresses, browser
							information, device identifiers
						</li>
						<li>
							<strong>Performance Metrics:</strong> System performance, latency
							measurements, provider statistics
						</li>
						<li>
							<strong>Log Data:</strong> Server logs, error reports, security
							events
						</li>
						<li>
							<strong>Cookies:</strong> Authentication tokens, preferences,
							session data
						</li>
					</ul>

					<h3 className="mt-6 mb-3 font-medium text-xl">
						2.3 Third-Party Provider Data
					</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>Provider Responses:</strong> AI-generated content from
							OpenAI, Anthropic, Google, Meta, Groq, DeepSeek, Cohere, Mistral,
							HuggingFace, and others
						</li>
						<li>
							<strong>Provider Metadata:</strong> Model information, usage
							statistics, cost calculations
						</li>
						<li>
							<strong>Provider Configurations:</strong> Your custom provider
							settings and API configurations
						</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						3. How We Use Your Information
					</h2>
					<p>We use your information to:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>
							<strong>Provide the Service:</strong> Route requests, manage
							conversations, process responses
						</li>
						<li>
							<strong>Account Management:</strong> Create accounts, authenticate
							users, manage subscriptions
						</li>
						<li>
							<strong>Analytics:</strong> Generate usage reports, cost analysis,
							performance metrics
						</li>
						<li>
							<strong>Optimization:</strong> Improve service performance, model
							selection, routing efficiency
						</li>
						<li>
							<strong>Support:</strong> Respond to inquiries, troubleshoot
							issues, provide assistance
						</li>
						<li>
							<strong>Security:</strong> Monitor for abuse, prevent fraud,
							maintain system security
						</li>
						<li>
							<strong>Compliance:</strong> Meet legal obligations and industry
							standards
						</li>
						<li>
							<strong>Communications:</strong> Send service notifications,
							updates, billing information
						</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						4. Third-Party Provider Data Sharing
					</h2>
					<div className="mb-4 rounded-lg border border-orange-200 bg-orange-50 p-4 dark:border-orange-800 dark:bg-orange-900/20">
						<p className="font-semibold text-orange-800 dark:text-orange-200">
							Important:
						</p>
						<p className="mt-1 text-orange-700 dark:text-orange-300">
							When you use our service, your prompts and data are sent to
							third-party AI providers to generate responses. Each provider may
							have different data handling practices.
						</p>
					</div>

					<h3 className="mb-3 font-medium text-xl">
						4.1 Providers We Work With
					</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>OpenAI:</strong> GPT models and related services
						</li>
						<li>
							<strong>Anthropic:</strong> Claude models and AI safety research
						</li>
						<li>
							<strong>Google AI:</strong> Gemini models and Google's AI services
						</li>
						<li>
							<strong>Meta:</strong> Llama models and AI research platforms
						</li>
						<li>
							<strong>Groq:</strong> High-performance AI inference
						</li>
						<li>
							<strong>DeepSeek:</strong> Advanced reasoning models
						</li>
						<li>
							<strong>Cohere:</strong> Enterprise language models
						</li>
						<li>
							<strong>Mistral:</strong> Open and commercial language models
						</li>
						<li>
							<strong>HuggingFace:</strong> Open-source model hosting
						</li>
						<li>
							<strong>Other Providers:</strong> Additional AI service providers
							as we expand
						</li>
					</ul>

					<h3 className="mt-6 mb-3 font-medium text-xl">
						4.2 Data Shared with Providers
					</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>Your prompts and conversation messages</li>
						<li>Model configuration parameters</li>
						<li>Request metadata (timestamp, model selection)</li>
						<li>Authentication tokens (when using your own API keys)</li>
					</ul>

					<h3 className="mt-6 mb-3 font-medium text-xl">
						4.3 Provider Data Policies
					</h3>
					<p>
						Each provider has their own data handling practices. We recommend
						reviewing their privacy policies:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>
							Some providers may use your data to improve their models unless
							you opt out
						</li>
						<li>Data retention periods vary by provider</li>
						<li>
							Some providers offer enterprise agreements with enhanced privacy
							protections
						</li>
						<li>Geographic data processing locations may differ</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						5. Data Storage and Security
					</h2>

					<h3 className="mb-3 font-medium text-xl">5.1 Data Storage</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>Database:</strong> PostgreSQL with encrypted sensitive
							fields
						</li>
						<li>
							<strong>Cache:</strong> Redis for temporary session and
							performance data
						</li>
						<li>
							<strong>Backups:</strong> Regular encrypted backups with retention
							policies
						</li>
						<li>
							<strong>Geographic Location:</strong> Data stored in secure cloud
							infrastructure
						</li>
					</ul>

					<h3 className="mt-6 mb-3 font-medium text-xl">
						5.2 Security Measures
					</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>Encryption:</strong> TLS for data in transit, encryption
							at rest for sensitive data
						</li>
						<li>
							<strong>Access Controls:</strong> Role-based access with principle
							of least privilege
						</li>
						<li>
							<strong>Authentication:</strong> Secure authentication via Clerk,
							API key management
						</li>
						<li>
							<strong>Monitoring:</strong> Real-time security monitoring and
							alerting
						</li>
						<li>
							<strong>Audit Logs:</strong> Comprehensive logging of system
							access and changes
						</li>
						<li>
							<strong>Regular Updates:</strong> Security patches and
							vulnerability management
						</li>
					</ul>

					<h3 className="mt-6 mb-3 font-medium text-xl">5.3 Data Retention</h3>
					<ul className="list-disc space-y-1 pl-6">
						<li>
							<strong>Chat History:</strong> Stored until you delete or account
							termination
						</li>
						<li>
							<strong>Usage Analytics:</strong> Retained for 2 years for billing
							and compliance
						</li>
						<li>
							<strong>Account Data:</strong> Retained while account is active
						</li>
						<li>
							<strong>Support Data:</strong> Retained for 3 years after
							resolution
						</li>
						<li>
							<strong>Log Data:</strong> Automatically deleted after 90 days
						</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						6. Data Subject Rights
					</h2>
					<p>You have the following rights regarding your personal data:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>
							<strong>Access:</strong> Request information about data we process
						</li>
						<li>
							<strong>Rectification:</strong> Correct inaccurate personal data
						</li>
						<li>
							<strong>Erasure:</strong> Request deletion of your personal data
						</li>
						<li>
							<strong>Portability:</strong> Receive your data in a structured
							format
						</li>
						<li>
							<strong>Restriction:</strong> Limit how we process your data
						</li>
						<li>
							<strong>Objection:</strong> Object to processing for legitimate
							interests
						</li>
						<li>
							<strong>Withdraw Consent:</strong> Withdraw consent where
							applicable
						</li>
					</ul>
					<p className="mt-4">
						To exercise these rights, contact us through our support channels.
						We will respond within 30 days.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						7. Cookies and Tracking
					</h2>
					<p>We use cookies and similar technologies for:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>
							<strong>Essential Cookies:</strong> Authentication, security,
							basic functionality
						</li>
						<li>
							<strong>Analytics Cookies:</strong> Usage statistics and
							performance monitoring
						</li>
						<li>
							<strong>Preference Cookies:</strong> User settings and interface
							preferences
						</li>
					</ul>
					<p className="mt-4">
						You can control cookies through your browser settings, but this may
						affect service functionality.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						8. International Data Transfers
					</h2>
					<p>
						Your data may be transferred to and processed in countries other
						than your country of residence. When we transfer data
						internationally, we use appropriate safeguards such as:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Standard contractual clauses</li>
						<li>Adequacy decisions by relevant authorities</li>
						<li>Certification schemes</li>
						<li>Provider-specific privacy frameworks</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">9. Children's Privacy</h2>
					<p>
						Our service is not intended for children under 13 (or the applicable
						age of digital consent in your jurisdiction). We do not knowingly
						collect personal information from children. If we become aware that
						we have collected data from a child, we will take steps to delete it
						promptly.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						10. Business Transfers
					</h2>
					<p>
						In the event of a merger, acquisition, or sale of assets, your
						information may be transferred to the new entity. We will provide
						notice before your data is transferred and becomes subject to
						different privacy practices.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">11. Legal Disclosure</h2>
					<p>We may disclose your information when required by law or to:</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Comply with legal process or government requests</li>
						<li>Protect our rights, property, or safety</li>
						<li>Protect users or the public from harm</li>
						<li>Investigate fraud or security issues</li>
						<li>Enforce our terms of service</li>
					</ul>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						12. Privacy Policy Updates
					</h2>
					<p>
						We may update this Privacy Policy periodically. Material changes
						will be communicated via:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Email notification to registered users</li>
						<li>Platform notifications</li>
						<li>Updated "Last modified" date on this page</li>
					</ul>
					<p className="mt-4">
						Continued use of the service after changes constitutes acceptance of
						the updated policy.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">13. Third-Party Links</h2>
					<p>
						Our service may contain links to third-party websites or services.
						We are not responsible for the privacy practices of these external
						sites. We encourage you to review their privacy policies before
						providing any personal information.
					</p>
				</section>

				<section>
					<h2 className="mb-4 font-semibold text-2xl">
						14. Contact Information
					</h2>
					<p>
						For privacy-related questions, concerns, or requests, please contact
						us through:
					</p>
					<ul className="mt-2 list-disc space-y-1 pl-6">
						<li>Our support channels available on the platform</li>
						<li>The contact information provided in your account dashboard</li>
						<li>Email support with "Privacy" in the subject line</li>
					</ul>
					<p className="mt-4">
						We are committed to addressing your privacy concerns promptly and
						transparently.
					</p>
				</section>
			</div>
		</div>
	);
}
