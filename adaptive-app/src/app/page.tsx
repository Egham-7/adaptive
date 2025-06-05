import HeroSection from "./_components/landing_page/hero";
import Pricing from "./_components/landing_page/pricing";
import Header from "./_components/landing_page/header";
import Footer from "./_components/landing_page/footer";
import ContentSection from "./_components/landing_page/content";
import FeaturesSection from "./_components/landing_page/features";
import StatsSection from "./_components/landing_page/stats-section";
import TestimonialsSection from "./_components/landing_page/testimonials-section";
import CallToAction from "./_components/landing_page/cta";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <main>
        <Header />
        <HeroSection />
        <FeaturesSection />
        <ContentSection />
        <StatsSection />
        <TestimonialsSection />
        <Pricing />
        <CallToAction />
        <Footer />
      </main>
    </div>
  );
}
