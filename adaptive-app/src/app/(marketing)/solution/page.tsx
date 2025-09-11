import type { Metadata } from "next";
import ChartsSection from "@/app/_components/landing_page/charts-section";
import ContentSection from "@/app/_components/landing_page/content";

export const metadata: Metadata = {
	title: "Solution | Adaptive - Enterprise AI Infrastructure",
	description:
		"Learn how Adaptive's AI routing solution reduces costs by 30-70% while improving reliability through intelligent model selection and multi-provider failover.",
	keywords: [
		"AI infrastructure",
		"cost reduction",
		"model routing",
		"enterprise AI",
		"multi-provider",
	],
	openGraph: {
		title: "Enterprise AI Solution - Cost Reduction & Reliability",
		description:
			"Reduce AI costs by 30-70% with intelligent model routing and multi-provider failover infrastructure.",
		type: "website",
	},
};

export default function SolutionPage() {
	return (
		<main>
			<article>
				<header className="sr-only">
					<h1>Adaptive AI Solution</h1>
				</header>
				<ContentSection />
				<ChartsSection />
			</article>
		</main>
	);
}
