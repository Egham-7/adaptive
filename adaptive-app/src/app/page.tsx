import { Suspense } from "react";
import ChartsSection from "./_components/landing_page/charts-section";
import ContentSection from "./_components/landing_page/content";
import CallToAction from "./_components/landing_page/cta";
import FeaturesSection from "./_components/landing_page/features";
import Footer from "./_components/landing_page/footer";
import Header from "./_components/landing_page/header";
import HeroSection from "./_components/landing_page/hero";
import Pricing from "./_components/landing_page/pricing";
import { WaitingListSection } from "./_components/landing_page/wating-list";

export default function LandingPage() {
	return (
		<Suspense>
			<Header />
			<HeroSection />
			<Pricing />
			<ChartsSection />
			<FeaturesSection />
			<ContentSection />
			<WaitingListSection />
			<CallToAction />
			<Footer />
		</Suspense>
	);
}
