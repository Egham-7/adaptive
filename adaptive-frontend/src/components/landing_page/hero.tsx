import { Button } from "@/components/ui/button";
import { CodeComparison } from "../magicui/code-comparison";
import { SignedIn, SignedOut, SignUpButton } from "@clerk/nextjs";
import { ArrowRight, Rocket } from "lucide-react";
import Link from "next/link";

const beforeCode = `from openai import OpenAI

openai = OpenAI(
    api_key="sk-your-api-key", 
    base_url="https://api.openai.com/v1"
)

async def generate_text(prompt: str) -> str:
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
`;

const afterCode = `from openai import OpenAI

openai = OpenAI(
    api_key="sk-your-api-key", 
    base_url="https://api.adaptive.com/v1" 
)

async def generate_text(prompt: str) -> str:
    response = await openai.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
`;

// When using these with your CodeComparison component:
// <CodeComparison
//   beforeCode={beforeCode}
//   afterCode={afterCode}
//   language="python" // Make sure to set language to "python"
//   filename="main.py"
//   lightTheme="github-light"
//   darkTheme="github-dark"
// />

export default function HeroSection() {
  return (
    <>
      <main className="overflow-hidden">
        <section>
          <div className="relative pt-24">
            <div className="mx-auto max-w-7xl px-6">
              <div className="max-w-3xl text-center sm:mx-auto lg:mr-auto lg:mt-0 lg:w-4/5">
                <Link
                  href="/"
                  className="rounded-md mx-auto flex w-fit items-center gap-2 border p-1 pr-3"
                >
                  <span className="bg-muted rounded-sm px-2 py-1 text-xs font-medium">
                    New
                  </span>
                  <span className="text-sm font-medium">
                    Adaptive LLM Infrastructure
                  </span>
                  <span className="block h-4 w-px bg-border"></span>
                  <ArrowRight className="size-4" />
                </Link>
                <h1 className="mt-8 text-balance text-4xl font-semibold font-display md:text-5xl xl:text-6xl xl:[line-height:1.125]">
                  Supercharge Your LLM Workloads
                </h1>
                <p className="mx-auto mt-8 hidden max-w-2xl text-wrap text-lg text-muted-foreground sm:block">
                  Optimize performance and cut costs with Adaptive's LLM-driven
                  infrastructure. Maximize efficiency and unleash the full
                  potential of your workloads, all while saving valuable
                  resources.
                </p>
                <p className="mx-auto mt-6 max-w-2xl text-wrap text-muted-foreground sm:hidden">
                  Optimize performance and cut costs with Adaptive's LLM-driven
                  infrastructure.
                </p>
                <div className="mt-8 flex justify-center gap-4">
                  {/* Show Sign Up button for unauthenticated users */}
                  <SignedOut>
                    <SignUpButton mode="modal" signInForceRedirectUrl="/home">
                      <Button
                        size="lg"
                        className="bg-linear-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle"
                      >
                        <Rocket className="relative size-4 mr-2" />
                        <span className="text-nowrap">Get Started</span>
                      </Button>
                    </SignUpButton>
                  </SignedOut>
                  {/* Show direct link to app for authenticated users */}
                  <SignedIn>
                    <Button
                      size="lg"
                      className="bg-linear-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle"
                      asChild
                    >
                      <Link href="/home">
                        <Rocket className="relative size-4 mr-2" />
                        <span className="text-nowrap">Go to Dashboard</span>
                      </Link>
                    </Button>
                  </SignedIn>
                  <Button
                    variant="outline"
                    size="lg"
                    className="border-primary-600 text-primary-600 hover:bg-primary-50/50 dark:border-primary-500 dark:text-primary-500 dark:hover:bg-primary-900/20"
                    asChild
                  >
                    <a href="#features">
                      <span className="text-nowrap">Learn More</span>
                    </a>
                  </Button>
                </div>
              </div>
            </div>
            <div className="relative mt-16 mb-12">
              <div className="relative mx-auto max-w-4xl px-6">
                <CodeComparison
                  beforeCode={beforeCode}
                  afterCode={afterCode}
                  language="python"
                  filename="llm.py"
                  lightTheme="github-light"
                  darkTheme="github-dark"
                />
              </div>
            </div>
          </div>
        </section>
        <section className="bg-background relative z-10 pb-16">
          <div className="m-auto max-w-5xl px-6">
            <h2 className="text-center text-lg font-medium">
              Your favorite companies are our partners.
            </h2>
            <div className="mx-auto mt-12 flex max-w-4xl flex-wrap items-center justify-center gap-x-12 gap-y-8 sm:gap-x-16 sm:gap-y-12">
              <img
                className="h-5 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/nvidia.svg"
                alt="Nvidia Logo"
                height="20"
                width="auto"
              />
              <img
                className="h-4 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/column.svg"
                alt="Column Logo"
                height="16"
                width="auto"
              />
              <img
                className="h-4 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/github.svg"
                alt="GitHub Logo"
                height="16"
                width="auto"
              />
              <img
                className="h-5 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/nike.svg"
                alt="Nike Logo"
                height="20"
                width="auto"
              />
              <img
                className="h-4 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/laravel.svg"
                alt="Laravel Logo"
                height="16"
                width="auto"
              />
              <img
                className="h-7 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/lilly.svg"
                alt="Lilly Logo"
                height="28"
                width="auto"
              />
              <img
                className="h-5 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/lemonsqueezy.svg"
                alt="Lemon Squeezy Logo"
                height="20"
                width="auto"
              />
              <img
                className="h-6 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/openai.svg"
                alt="OpenAI Logo"
                height="24"
                width="auto"
              />
              <img
                className="h-4 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/tailwindcss.svg"
                alt="Tailwind CSS Logo"
                height="16"
                width="auto"
              />
              <img
                className="h-5 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/vercel.svg"
                alt="Vercel Logo"
                height="20"
                width="auto"
              />
              <img
                className="h-5 w-fit dark:invert"
                src="https://html.tailus.io/blocks/customers/zapier.svg"
                alt="Zapier Logo"
                height="20"
                width="auto"
              />
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
