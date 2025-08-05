import type { Metadata } from "next";
import Pricing from "@/app/_components/landing_page/pricing";

export const metadata: Metadata = {
	title: "Pricing | Adaptive - Simple & Transparent AI API Pricing",
	description:
		"Explore Adaptive's transparent pricing plans. From pay-as-you-go Developer plans to Enterprise solutions. Start with $10 free credit.",
	keywords: [
		"API pricing",
		"AI costs",
		"pay-as-you-go",
		"enterprise pricing",
		"free credit",
	],
	openGraph: {
		title: "Simple AI API Pricing - Start Free",
		description:
			"Transparent pricing for AI model routing. Pay-as-you-go or fixed monthly plans. Start with $10 free credit.",
		type: "website",
	},
};

export default function PricingPage() {
	return (
		<main>
			<article>
				<Pricing />
			</article>
		</main>
	);
}
