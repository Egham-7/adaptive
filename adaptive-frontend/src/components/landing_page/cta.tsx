import { Button } from "@/components/ui/button";

export default function CTA() {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-primary-600 text-white">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
              Smarter AI, Lower Costs.
            </h2>
            <p className="mx-auto max-w-[700px] text-primary-100 md:text-xl">
              Dynamically route queries to the most efficient AI models,
              reducing costs without sacrificing performance.
            </p>
          </div>
          <div className="space-x-4">
            <Button
              size="lg"
              className="bg-secondary-500 hover:bg-secondary-600 text-white"
            >
              Try It for Free
            </Button>
            <Button
              size="lg"
              className="bg-transparent border-2 border-primary-100 text-white hover:bg-primary-700"
            >
              Learn More
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
