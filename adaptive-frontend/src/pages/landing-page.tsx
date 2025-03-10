import Hero from "@/components/landing_page/hero";
import Features from "@/components/landing_page/features";
import Pricing from "@/components/landing_page/pricing";
import CTA from "@/components/landing_page/cta";
import Header from "@/components/landing_page/header";
import Footer from "@/components/landing_page/footer";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <main>
        <Header />
        <Hero />
        <Features />
        <Pricing />
        <CTA />
        <Footer />
      </main>
    </div>
  );
}
