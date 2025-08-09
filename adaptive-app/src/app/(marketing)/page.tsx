import type { Metadata } from "next";
import { Suspense } from "react";
import ChartsSection from "@/app/_components/landing_page/charts-section";
import ContentSection from "@/app/_components/landing_page/content";
import CallToAction from "@/app/_components/landing_page/cta";
import FeaturesSection from "@/app/_components/landing_page/features";
import HeroSection from "@/app/_components/landing_page/hero";
import Pricing from "@/app/_components/landing_page/pricing";

export const metadata: Metadata = {
	title: "Adaptive - Intelligent AI Model Routing & Cost Optimization",
	description:
		"Reduce AI costs by 30-70% with intelligent model routing. Automatic failover, smart request handling, and multi-provider support. Start free with $10 credit.",
	keywords: [
		"AI model routing",
		"cost optimization",
		"OpenAI alternative",
		"multi-provider AI",
		"intelligent routing",
	],
	openGraph: {
		title: "Adaptive - Smart AI Model Routing",
		description:
			"Reduce AI costs by 30-70% with intelligent model routing and automatic failover. Start free today.",
		type: "website",
	},
};

export default function LandingPage() {
	return (
		<Suspense>
			<main>
				<HeroSection />
				<ChartsSection />
				<FeaturesSection />
				<ContentSection />
				<Pricing />
				<CallToAction />
			</main>
		</Suspense>
	);
}
