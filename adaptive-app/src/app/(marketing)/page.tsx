import { Suspense } from "react";
import ChartsSection from "@/app/_components/landing_page/charts-section";
import ContentSection from "@/app/_components/landing_page/content";
import CallToAction from "@/app/_components/landing_page/cta";
import FeaturesSection from "@/app/_components/landing_page/features";
import FooterSection from "@/app/_components/landing_page/footer";
import GetStartedSection from "@/app/_components/landing_page/get-started-section";
import HeroSection from "@/app/_components/landing_page/hero";
import Pricing from "@/app/_components/landing_page/pricing";

export default function LandingPage() {
	return (
		<Suspense>
			<main>
				<HeroSection />
				<ChartsSection />
				<GetStartedSection />
				<FeaturesSection />
				<ContentSection />
				<Pricing />
				<CallToAction />
				<FooterSection />
			</main>
		</Suspense>
	);
}
