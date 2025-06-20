"use client";
import { CodeComparison } from "@/components/magicui/code-comparison";
import { Button } from "@/components/ui/button";
import { SignUpButton, SignedIn, SignedOut } from "@clerk/nextjs";
import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, Rocket } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import ApiButton from "./new_infrastructure_button";

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
function FloatingPaths({ position }: { position: number }) {
	const [hasCompletedInitialPass, setHasCompletedInitialPass] = useState(false);
	const { scrollY } = useScroll();
	const scrollProgress = useTransform(scrollY, [0, 1000], [1, 0]); // Reversed: scroll down = forward, scroll up = backward

	const paths = Array.from({ length: 36 }, (_, i) => ({
		id: i,
		d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
			380 - i * 5 * position
		} -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
			152 - i * 5 * position
		} ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
			684 - i * 5 * position
		} ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
		color: `rgba(15,23,42,${0.1 + i * 0.03})`,
		width: 0.5 + i * 0.03,
	}));

	return (
		<div className="pointer-events-none absolute inset-0">
			<svg
				className="h-full w-full text-slate-950 dark:text-white"
				viewBox="0 0 696 316"
				fill="none"
			>
				<title>Background Paths</title>
				{paths.map((path) => (
					<motion.path
						key={path.id}
						d={path.d}
						stroke="currentColor"
						strokeWidth={path.width}
						strokeOpacity={0.1 + path.id * 0.03}
						initial={{ pathLength: 0.3, opacity: 0.6 }}
						animate={
							hasCompletedInitialPass
								? {} // No animation, just use style for scroll control
								: {
										pathLength: 1,
										opacity: [0.3, 0.6, 0.3],
										pathOffset: [0, 1],
									}
						}
						style={
							hasCompletedInitialPass
								? {
										pathOffset: scrollProgress, // Scroll controls pathOffset directly
									}
								: {}
						}
						transition={
							hasCompletedInitialPass
								? {}
								: {
										duration: 20 + Math.random() * 5,
										delay: path.id * 0.1,
										ease: "linear",
										onComplete: () => {
											if (path.id === paths.length - 1) {
												setHasCompletedInitialPass(true);
											}
										},
									}
						}
					/>
				))}
			</svg>
		</div>
	);
}

export default function HeroSection() {
	return (
		<>
			<section className="relative flex min-h-screen items-center overflow-hidden bg-white dark:bg-neutral-950">
				{/* Background Paths */}
				<div className="absolute inset-0">
					<FloatingPaths position={1} />
					<FloatingPaths position={-1} />
				</div>

				<div className="relative z-10 w-full py-12">
					<div className="mx-auto max-w-7xl px-6">
						<div className="max-w-4xl text-center sm:mx-auto lg:mt-0 lg:mr-auto lg:w-full">
							<ApiButton />
							<h1 className="mt-8 text-balance font-display font-semibold text-4xl md:text-5xl xl:text-7xl xl:[line-height:1.125]">
								{"We Provide the Best LLM Inference".split(" ").map((word) => (
									<span key={word} className="mr-2 inline-block last:mr-0">
										{word.split("").map((letter, letterIndex) => (
											<motion.span
												key={word.slice(0, letterIndex + 1)}
												initial={{ y: 100, opacity: 0 }}
												animate={{ y: 0, opacity: 1 }}
												transition={{
													delay: letterIndex * 0.03,
													type: "spring",
													stiffness: 150,
													damping: 25,
												}}
												className="inline-block bg-gradient-to-r from-neutral-900 to-neutral-700/80 bg-clip-text text-transparent dark:from-white dark:to-white/80"
											>
												{letter}
											</motion.span>
										))}
									</span>
								))}
							</h1>

							<p className="mx-auto mt-8 hidden max-w-3xl text-wrap text-muted-foreground sm:block">
								Optimize performance and cut costs with Adaptive's LLM-driven
								infrastructure. Maximize efficiency and unleash the full
								potential of your workloads, all while saving valuable
								resources.
							</p>
							<p className="mx-auto mt-6 max-w-3xl text-wrap text-muted-foreground sm:hidden">
								Optimize performance and cut costs with Adaptive's LLM-driven
								infrastructure.
							</p>

							<div className="mt-8 flex justify-center gap-4">
								<SignedOut>
									<SignUpButton signInForceRedirectUrl="/chat-platform">
										<Button
											size="lg"
											className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
										>
											<Rocket className="relative mr-2 size-4" />
											<span className="text-nowrap">Get Started</span>
										</Button>
									</SignUpButton>
								</SignedOut>
								<SignedIn>
									<Button
										size="lg"
										className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
										asChild
									>
										<Link href="/chat-platform">
											<Rocket className="relative mr-2 size-4" />
											<span className="text-nowrap">Go to Dashboard</span>
										</Link>
									</Button>
								</SignedIn>
								<Button
									variant="outline"
									size="lg"
									className="border-primary text-primary hover:bg-primary/10 dark:hover:bg-primary/20"
									asChild
								>
									<a href="#features">
										<span className="text-nowrap">Learn More</span>
									</a>
								</Button>
							</div>
						</div>
					</div>
					{/* Code comparison section */}
					<div className="relative mt-16 mb-12">
						<div className="mx-auto max-w-4xl px-6">
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
		</>
	);
}
