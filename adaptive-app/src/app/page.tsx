import ContentSection from "./_components/landing_page/content";
import CallToAction from "./_components/landing_page/cta";
import FeaturesSection from "./_components/landing_page/features";
import Footer from "./_components/landing_page/footer";
import Header from "./_components/landing_page/header";
import HeroSection from "./_components/landing_page/hero";
import Pricing from "./_components/landing_page/pricing";
import StatsSection from "./_components/landing_page/stats-section";
import TestimonialsSection from "./_components/landing_page/testimonials-section";

export default function LandingPage() {
	return (
		<>
			<Header />
			<HeroSection />
			<FeaturesSection />
			<ContentSection />
			<StatsSection />
			<TestimonialsSection />
			<Pricing />
			<CallToAction />
			<Footer />
		</>
	);
}
