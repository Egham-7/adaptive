import { Suspense } from "react";
import ChartsSection from "./_components/landing_page/charts-section";
import ContentSection from "./_components/landing_page/content";
import CallToAction from "./_components/landing_page/cta";
import FeaturesSection from "./_components/landing_page/features";
import Header from "./_components/landing_page/header";
import HeroSection from "./_components/landing_page/hero";
import Pricing from "./_components/landing_page/pricing";

export default function LandingPage() {
	return (
		<Suspense>
			<Header />
			<HeroSection />
			<ChartsSection />
			<FeaturesSection />
			<ContentSection />
			<Pricing />
			<CallToAction />
		</Suspense>
	);
}
