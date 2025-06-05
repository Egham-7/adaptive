import HeroSection from "@/components/landing_page/hero";
import Pricing from "@/components/landing_page/pricing";
import CallToAction from "@/components/landing_page/cta";
import Header from "@/components/landing_page/header";
import Footer from "@/components/landing_page/footer";
import ContentSection from "@/components/landing_page/content";
import FeaturesSection from "@/components/landing_page/features";
import StatsSection from "@/components/landing_page/stats-section";
import TestimonialsSection from "@/components/landing_page/testimonials-section";

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
