import type { Metadata } from "next";
import Customers from "@/app/_components/landing_page/customers";
import TestimonialsSection from "@/app/_components/landing_page/testimonials-section";

export const metadata: Metadata = {
	title: "About | Adaptive - Trusted by Leading Companies",
	description:
		"Learn about Adaptive's mission to optimize AI infrastructure. Trusted by leading companies for reliable, cost-effective AI model routing and optimization.",
	keywords: [
		"about Adaptive",
		"AI company",
		"testimonials",
		"customers",
		"mission",
	],
	openGraph: {
		title: "About Adaptive - Trusted AI Infrastructure",
		description:
			"Learn about our mission to optimize AI infrastructure for leading companies worldwide.",
		type: "website",
	},
};

export default function AboutPage() {
	return (
		<main>
			<article>
				<header className="container mx-auto px-6 py-16 text-center">
					<h1 className="font-display font-semibold text-4xl lg:text-5xl">
						About Adaptive
					</h1>
					<p className="mx-auto mt-4 max-w-2xl text-lg text-muted-foreground">
						We're building the future of AI infrastructure, making intelligent
						model routing accessible to every developer and organization.
					</p>
				</header>
				<TestimonialsSection />
				<Customers />
			</article>
		</main>
	);
}
