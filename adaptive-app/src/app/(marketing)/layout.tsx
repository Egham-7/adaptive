import type { Metadata } from "next";
import { Suspense } from "react";
import FloatingHeader from "@/app/_components/landing_page/floating-header";
import Header from "@/app/_components/landing_page/header";

export const metadata: Metadata = {
	title: "LLM Adaptive - Intelligent AI Model Router | 60% Cost Reduction",
	description:
		"The first LLM router that thinks. Intelligent prompt analysis automatically selects optimal models. Sub-millisecond routing, automatic failover, format adaptation. Free $10 credit.",
	keywords: [
		"LLM router",
		"AI model routing",
		"intelligent prompt analysis",
		"LLM cost optimization",
		"OpenAI router",
		"Anthropic router",
		"multi-provider LLM",
		"AI infrastructure",
		"LLM fallback",
		"smart AI routing",
		"LLM gateway",
		"AI cost reduction",
	],
	openGraph: {
		title: "LLM Adaptive - First Intelligent AI Model Router",
		description:
			"Reduce LLM costs by 60% with intelligent prompt analysis and automatic model selection. Zero-config deployment with sub-millisecond routing.",
		type: "website",
	},
};

export default function MarketingLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<Suspense>
			<Header />
			<FloatingHeader />
			{children}
		</Suspense>
	);
}
