import { Button } from "@/components/ui/button";
import { Rocket } from "lucide-react";
import Link from "next/link";
import { SignedOut, SignUpButton, SignedIn } from "@clerk/nextjs";

export default function CallToAction() {
  return (
    <section className="py-16 md:py-32">
      <div className="mx-auto max-w-5xl px-6">
        <div className="text-center">
          <h2 className="text-balance font-display text-4xl font-semibold lg:text-5xl">
            Start Building with Adaptive
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
            Optimize your LLM workloads today and experience the power of
            intelligent infrastructure that scales with your needs.
          </p>
          <div className="mt-12 flex flex-wrap justify-center gap-4">
            <SignedOut>
              <SignUpButton mode="modal">
                <Button
                  size="lg"
                  className="font-medium shadow-subtle transition-opacity hover:opacity-90 bg-primary text-primary-foreground"
                >
                  <Rocket className="relative mr-2 size-4" />
                  <span>Get Started</span>
                </Button>
              </SignUpButton>
            </SignedOut>

            <SignedIn>
              <Button
                size="lg"
                className="font-medium shadow-subtle transition-opacity hover:opacity-90 bg-primary text-primary-foreground"
              >
                <Rocket className="relative mr-2 size-4" />
                <Link href="/home">Get Started</Link>
              </Button>
            </SignedIn>

            <Button
              asChild
              size="lg"
              variant="outline"
              className="border-primary text-primary hover:bg-primary/10 dark:hover:bg-primary/20"
            >
              <Link href="/">
                <span>Book Demo</span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
