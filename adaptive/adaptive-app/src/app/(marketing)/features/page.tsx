import type { Metadata } from "next";
import FeaturesSection from "@/app/_components/landing_page/features";

export const metadata: Metadata = {
	title: "Features | Adaptive - AI Model Routing & Optimization",
	description:
		"Discover Adaptive's intelligent AI features: smart request handling, reliable failover, cost optimization, and performance analytics for your applications.",
	keywords: [
		"AI routing",
		"model optimization",
		"cost savings",
		"failover",
		"performance analytics",
	],
	openGraph: {
		title: "AI Features - Smart Routing & Optimization",
		description:
			"Intelligent AI optimization platform with smart request handling, reliable failover, and cost optimization features.",
		type: "website",
	},
};

export default function FeaturesPage() {
	return (
		<main>
			<article>
				<FeaturesSection />
			</article>
		</main>
	);
}
