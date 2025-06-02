import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Rocket } from "lucide-react";
import { SignedOut, SignUpButton, SignedIn } from "@clerk/clerk-react";

export default function CallToAction() {
  return (
    <section className="py-16 md:py-32">
      <div className="mx-auto max-w-5xl px-6">
        <div className="text-center">
          <h2 className="text-balance text-4xl font-semibold font-display lg:text-5xl">
            Start Building with Adaptive
          </h2>
          <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
            Optimize your LLM workloads today and experience the power of
            intelligent infrastructure that scales with your needs.
          </p>
          <div className="mt-12 flex flex-wrap justify-center gap-4">
            <SignedOut>
              <SignUpButton mode="modal">
                <Button
                  size="lg"
                  className="bg-linear-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle"
                >
                  <Rocket className="relative size-4 mr-2" />
                  <span>Get Started</span>
                </Button>
              </SignUpButton>
            </SignedOut>

            <SignedIn>
              <Button
                size="lg"
                className="bg-linear-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle"
              >
                <Rocket className="relative size-4 mr-2" />
                <Link to="/home">Get Started</Link>
              </Button>
            </SignedIn>

            <Button
              asChild
              size="lg"
              variant="outline"
              className="border-primary-600 text-primary-600 hover:bg-primary-50/50 dark:border-primary-500 dark:text-primary-500 dark:hover:bg-primary-900/20"
            >
              <Link to="/">
                <span>Book Demo</span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
