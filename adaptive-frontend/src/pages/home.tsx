import Hero from "@/components/landing_page/hero";
import Features from "@/components/landing_page/features";
import Pricing from "@/components/landing_page/pricing";
import CTA from "@/components/landing_page/cta";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <main>
        <Hero />
        <Features />
        <Pricing />
        <CTA />
      </main>
    </div>
  );
}
