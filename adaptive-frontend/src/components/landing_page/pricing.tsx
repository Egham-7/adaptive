import { Link } from "@tanstack/react-router";
import { Check, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { SignUpButton } from "@clerk/clerk-react";

export default function Pricing() {
  return (
    <section className="py-16 md:py-32 overflow-hidden">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mx-auto max-w-2xl space-y-6 text-center">
          <h1 className="text-center text-4xl font-semibold font-display lg:text-5xl">
            Pay Only for What You Use
          </h1>
          <p className="text-muted-foreground">
            Adaptive's usage-based pricing ensures you only pay for the
            resources you consume. Scale up or down instantly with no upfront
            commitments or hidden fees.
          </p>
        </div>

        <div className="mt-8 grid gap-6 md:mt-20 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="font-medium">Developer</CardTitle>
              <div className="flex items-baseline my-3">
                <span className="text-2xl font-semibold">$0.005</span>
                <span className="text-sm text-muted-foreground ml-1">
                  / 1K tokens
                </span>
              </div>
              <CardDescription className="text-sm">
                No minimum spend, pay as you go
              </CardDescription>
              <Button asChild variant="outline" className="mt-4 w-full">
                <SignUpButton></SignUpButton>
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              <hr className="border-dashed" />
              <ul className="list-outside space-y-3 text-sm">
                {[
                  "Pay-as-you-go pricing",
                  "Basic LLM routing",
                  "Standard API access",
                  "Community support",
                  "$10 free credit to start",
                  "No credit card required",
                ].map((item, index) => (
                  <li key={index} className="flex items-center gap-2">
                    <Check className="size-3 text-primary-600" />
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card className="relative">
            <span className="absolute inset-x-0 -top-3 mx-auto flex h-6 w-fit items-center rounded-full bg-linear-to-r from-primary-600 to-secondary-600 px-3 py-1 text-xs font-medium text-white ring-1 ring-inset ring-white/20 ring-offset-1 ring-offset-gray-950/5">
              Popular
            </span>
            <CardHeader>
              <CardTitle className="font-medium">Business</CardTitle>
              <div className="flex items-baseline my-3">
                <span className="text-2xl font-semibold">$0.004</span>
                <span className="text-sm text-muted-foreground ml-1">
                  / 1K tokens
                </span>
              </div>
              <CardDescription className="text-sm">
                $100 minimum monthly spend
              </CardDescription>
              <Button
                asChild
                className="mt-4 w-full bg-linear-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity shadow-subtle"
              >
                <SignUpButton>
                  <Button>
                    <Zap className="relative size-4 mr-2" />
                    <span>Get Started</span>
                  </Button>
                </SignUpButton>
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              <hr className="border-dashed" />
              <ul className="list-outside space-y-3 text-sm">
                {[
                  "20% volume discount",
                  "Advanced LLM routing",
                  "Multi-model fallbacks",
                  "Priority email support",
                  "Usage analytics dashboard",
                  "Custom API keys",
                  "Request caching",
                  "Webhook integrations",
                  "Team access controls",
                  "99.9% uptime SLA",
                ].map((item, index) => (
                  <li key={index} className="flex items-center gap-2">
                    <Check className="size-3 text-primary-600" />
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle className="font-medium">Enterprise</CardTitle>
              <div className="flex items-baseline my-3">
                <span className="text-2xl font-semibold">Custom</span>
              </div>
              <CardDescription className="text-sm">
                Volume-based discounts available
              </CardDescription>
              <Button
                asChild
                variant="outline"
                className="mt-4 w-full border-primary-600 text-primary-600 hover:bg-primary-50/50 dark:border-primary-500 dark:text-primary-500 dark:hover:bg-primary-900/20"
              >
                <Link to="/">Contact Sales</Link>
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              <hr className="border-dashed" />
              <ul className="list-outside space-y-3 text-sm">
                {[
                  "Volume-based pricing",
                  "Dedicated account manager",
                  "Custom SLAs and support",
                  "Advanced cost optimization",
                  "Private model deployment",
                  "Custom model fine-tuning",
                  "Enterprise SSO",
                  "Audit logs & compliance",
                  "On-premise deployment options",
                  "MSA and custom contracts",
                ].map((item, index) => (
                  <li key={index} className="flex items-center gap-2">
                    <Check className="size-3 text-primary-600" />
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>

        <div className="mt-16 mx-auto max-w-3xl">
          <div className="rounded-lg border p-6 bg-muted/30">
            <h3 className="text-lg font-medium mb-4">Volume Discounts</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-md bg-background border">
                <div className="text-sm font-medium text-muted-foreground">
                  1M - 10M tokens/month
                </div>
                <div className="mt-2 text-xl font-semibold">10% off</div>
              </div>
              <div className="p-4 rounded-md bg-background border">
                <div className="text-sm font-medium text-muted-foreground">
                  10M - 50M tokens/month
                </div>
                <div className="mt-2 text-xl font-semibold">20% off</div>
              </div>
              <div className="p-4 rounded-md bg-background border">
                <div className="text-sm font-medium text-muted-foreground">
                  50M+ tokens/month
                </div>
                <div className="mt-2 text-xl font-semibold">Contact us</div>
              </div>
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              All plans include access to our core infrastructure. Volume
              discounts are applied automatically based on your monthly usage.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
