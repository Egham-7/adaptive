import { Button } from "@/components/ui/button";

export default function Hero() {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48 bg-background">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl text-primary-600 dark:text-primary-200">
              Supercharge Your LLM Workloads
            </h1>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl dark:text-muted-foreground">
              Optimize performance and cut costs with Adaptive's LLM-driven
              infrastructure. Maximize efficiency and unleash the full potential
              of your workloads, all while saving valuable resources.
            </p>
          </div>
          <div className="space-x-4">
            <Button
              size="lg"
              className="bg-primary-600 text-primary-50 hover:bg-primary-700 dark:bg-primary-500 dark:text-primary-50 dark:hover:bg-primary-600"
            >
              Get Started
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="border-primary-600 text-primary-600 hover:bg-primary-100 dark:border-primary-500 dark:text-primary-500 dark:hover:bg-primary-600 dark:hover:text-primary-50"
            >
              Learn More
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
