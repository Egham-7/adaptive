"use client";

import { SignedIn, SignedOut, SignUpButton } from "@clerk/nextjs";
import { motion } from "framer-motion";
import { MessageSquare, Rocket, Zap } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { forwardRef, useCallback, useRef, useState } from "react";
import { AnimatedBeam } from "@/components/ui/animated-beam";
import { Button } from "@/components/ui/button";
import { TextRotate } from "@/components/ui/text-rotate";
import { cn } from "@/lib/utils";

interface CircleProps {
	className?: string;
	children?: React.ReactNode;
}

interface CircleState {
	prompt: boolean;
	adaptive: boolean;
	openai: boolean;
	anthropic: boolean;
	google: boolean;
	meta: boolean;
}

const Circle = forwardRef<HTMLDivElement, CircleProps>(
	({ className, children }, ref) => {
		return (
			<div
				ref={ref}
				className={cn(
					"z-10 flex size-16 items-center justify-center rounded-full border-2 bg-white/90 p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)] backdrop-blur-sm dark:bg-neutral-800/90",
					className,
				)}
			>
				{children}
			</div>
		);
	},
);

Circle.displayName = "Circle";

const PromptCircle = forwardRef<HTMLDivElement, { className?: string }>(
	({ className }, ref) => (
		<div
			ref={ref}
			className={cn(
				"z-10 flex size-20 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-purple-600 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]",
				className,
			)}
		>
			<MessageSquare className="h-8 w-8 text-white" />
		</div>
	),
);

PromptCircle.displayName = "PromptCircle";

const AdaptiveCircle = forwardRef<HTMLDivElement, { className?: string }>(
	({ className }, ref) => (
		<div
			ref={ref}
			className={cn(
				"z-10 flex size-24 items-center justify-center rounded-full bg-gradient-to-r from-primary to-primary/80 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]",
				className,
			)}
		>
			<Zap className="h-10 w-10 text-white" />
		</div>
	),
);

AdaptiveCircle.displayName = "AdaptiveCircle";

export default function HeroSection() {
	// Rotating text options
	const rotatingTexts = [
		"Smart Model Routing",
		"Tailored AI",
		"Intelligent Model Selection",
	];

	// Refs for animated beams
	const containerRef = useRef<HTMLDivElement>(null);
	const promptRef = useRef<HTMLDivElement>(null);
	const adaptiveRef = useRef<HTMLDivElement>(null);
	const openaiRef = useRef<HTMLDivElement>(null);
	const anthropicRef = useRef<HTMLDivElement>(null);
	const googleRef = useRef<HTMLDivElement>(null);
	const metaRef = useRef<HTMLDivElement>(null);

	// State to track when circles are ready
	const [circlesReady, setCirclesReady] = useState<CircleState>({
		prompt: false,
		adaptive: false,
		openai: false,
		anthropic: false,
		google: false,
		meta: false,
	});

	// Callback to mark circles as ready
	const markCircleReady = useCallback((circleName: keyof CircleState) => {
		setCirclesReady((prev) => ({ ...prev, [circleName]: true }));
	}, []);

	// Check if specific beam paths should be rendered
	const shouldRenderBeam1 = circlesReady.prompt && circlesReady.adaptive;
	const shouldRenderBeam2 = circlesReady.adaptive && circlesReady.openai;
	const shouldRenderBeam3 = circlesReady.adaptive && circlesReady.anthropic;
	const shouldRenderBeam4 = circlesReady.adaptive && circlesReady.google;
	const shouldRenderBeam5 = circlesReady.adaptive && circlesReady.meta;

	return (
		<section className="relative flex min-h-screen items-center overflow-hidden bg-white dark:bg-neutral-950">
			<div className="relative z-10 w-full py-12">
				<div className="mx-auto max-w-7xl px-6">
					<div className="max-w-4xl text-center sm:mx-auto lg:mt-0 lg:mr-auto lg:w-full">
						<h1 className="mt-8 text-balance text-center font-display font-semibold text-4xl md:text-5xl xl:text-7xl xl:[line-height:1.125]">
							{"We Provide ".split(" ").map((word, wordIndex) => (
								<span key={word} className="mr-2 inline-block last:mr-0">
									{word.split("").map((letter, letterIndex) => (
										<motion.span
											key={word.slice(0, letterIndex + 1)}
											initial={{ y: 100, opacity: 0 }}
											animate={{ y: 0, opacity: 1 }}
											transition={{
												delay: (wordIndex * 3 + letterIndex) * 0.03,
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

							<TextRotate
								texts={rotatingTexts}
								rotationInterval={3000}
								staggerDuration={0.02}
								staggerFrom="first"
								splitBy="characters"
								mainClassName="bg-gradient-to-r flex items-center justify-center from-primary to-primary/80 bg-clip-text text-transparent"
								elementLevelClassName="inline-block"
								initial={{ y: 100, opacity: 0 }}
								animate={{ y: 0, opacity: 1 }}
								exit={{ y: -100, opacity: 0 }}
								transition={{
									type: "spring",
									stiffness: 150,
									damping: 25,
								}}
							/>
						</h1>

						<p className="mx-auto mt-8 max-w-3xl text-balance text-muted-foreground">
							Adaptive intelligently routes your queries to the best AI model
							for each task. Get optimal results while reducing costs by up to
							90% compared to using premium models for everything.
						</p>

						{/* Animated Beam Visualization */}
						<div
							className="relative mt-12 mb-8 flex h-[300px] w-full items-center justify-center overflow-hidden"
							ref={containerRef}
							role="img"
							aria-label="Adaptive AI infrastructure connecting user prompts to various AI providers"
						>
							<div className="flex w-full items-center justify-between px-4 sm:px-0">
								{/* Prompt Input */}
								<motion.div
									layout
									initial={{ opacity: 0, x: -50 }}
									animate={{ opacity: 1, x: 0 }}
									transition={{
										delay: 1,
										duration: 0.6,
										layout: { duration: 0.3 },
									}}
									onAnimationComplete={() => markCircleReady("prompt")}
								>
									<PromptCircle ref={promptRef} />
								</motion.div>

								{/* Adaptive Hub (Center) */}
								<motion.div
									layout
									initial={{ opacity: 0, scale: 0.8 }}
									animate={{ opacity: 1, scale: 1 }}
									transition={{
										delay: 1.5,
										duration: 0.6,
										layout: { duration: 0.3 },
									}}
									onAnimationComplete={() => markCircleReady("adaptive")}
								>
									<AdaptiveCircle ref={adaptiveRef} />
								</motion.div>

								{/* Provider Icons */}
								<div className="flex flex-col gap-3">
									<motion.div
										layout
										initial={{ opacity: 0, x: 50 }}
										animate={{ opacity: 1, x: 0 }}
										transition={{
											delay: 2,
											duration: 0.6,
											layout: { duration: 0.3 },
										}}
										onAnimationComplete={() => markCircleReady("openai")}
									>
										<Circle ref={openaiRef}>
											<Image
												src="/logos/openai.webp"
												alt="OpenAI"
												width={32}
												height={32}
												className="h-8 w-8 object-contain mix-blend-multiply dark:mix-blend-normal dark:invert"
											/>
										</Circle>
									</motion.div>
									<motion.div
										layout
										initial={{ opacity: 0, x: 50 }}
										animate={{ opacity: 1, x: 0 }}
										transition={{
											delay: 2.2,
											duration: 0.6,
											layout: { duration: 0.3 },
										}}
										onAnimationComplete={() => markCircleReady("anthropic")}
									>
										<Circle ref={anthropicRef}>
											<Image
												src="/logos/anthropic.jpeg"
												alt="Anthropic"
												width={32}
												height={32}
												className="h-8 w-8 rounded-sm object-contain"
											/>
										</Circle>
									</motion.div>
									<motion.div
										layout
										initial={{ opacity: 0, x: 50 }}
										animate={{ opacity: 1, x: 0 }}
										transition={{
											delay: 2.4,
											duration: 0.6,
											layout: { duration: 0.3 },
										}}
										onAnimationComplete={() => markCircleReady("google")}
									>
										<Circle ref={googleRef}>
											<Image
												src="/logos/google.svg"
												alt="Google"
												width={32}
												height={32}
												className="h-8 w-8 object-contain"
											/>
										</Circle>
									</motion.div>
									<motion.div
										layout
										initial={{ opacity: 0, x: 50 }}
										animate={{ opacity: 1, x: 0 }}
										transition={{
											delay: 2.6,
											duration: 0.6,
											layout: { duration: 0.3 },
										}}
										onAnimationComplete={() => markCircleReady("meta")}
									>
										<Circle ref={metaRef}>
											<Image
												src="/logos/meta.png"
												alt="Meta"
												width={32}
												height={32}
												className="h-8 w-8 object-contain"
											/>
										</Circle>
									</motion.div>
								</div>
							</div>

							{/* Animated Beams - Only render when circles are ready */}
							{shouldRenderBeam1 && (
								<AnimatedBeam
									containerRef={containerRef as React.RefObject<HTMLElement>}
									fromRef={promptRef as React.RefObject<HTMLElement>}
									toRef={adaptiveRef as React.RefObject<HTMLElement>}
									delay={0.2}
									duration={2}
									gradientStartColor="#3b82f6"
									gradientStopColor="#8b5cf6"
									startXOffset={40}
									endXOffset={-48}
								/>
							)}
							{shouldRenderBeam2 && (
								<AnimatedBeam
									containerRef={containerRef as React.RefObject<HTMLElement>}
									fromRef={adaptiveRef as React.RefObject<HTMLElement>}
									toRef={openaiRef as React.RefObject<HTMLElement>}
									delay={0.4}
									duration={1.8}
									curvature={-20}
									startXOffset={48}
									endXOffset={-32}
								/>
							)}
							{shouldRenderBeam3 && (
								<AnimatedBeam
									containerRef={containerRef as React.RefObject<HTMLElement>}
									fromRef={adaptiveRef as React.RefObject<HTMLElement>}
									toRef={anthropicRef as React.RefObject<HTMLElement>}
									delay={0.6}
									duration={1.6}
									curvature={-5}
									startXOffset={48}
									endXOffset={-32}
								/>
							)}
							{shouldRenderBeam4 && (
								<AnimatedBeam
									containerRef={containerRef as React.RefObject<HTMLElement>}
									fromRef={adaptiveRef as React.RefObject<HTMLElement>}
									toRef={googleRef as React.RefObject<HTMLElement>}
									delay={0.8}
									duration={1.4}
									curvature={5}
									startXOffset={48}
									endXOffset={-32}
								/>
							)}
							{shouldRenderBeam5 && (
								<AnimatedBeam
									containerRef={containerRef as React.RefObject<HTMLElement>}
									fromRef={adaptiveRef as React.RefObject<HTMLElement>}
									toRef={metaRef as React.RefObject<HTMLElement>}
									delay={1.0}
									duration={1.2}
									curvature={20}
									startXOffset={48}
									endXOffset={-32}
								/>
							)}
						</div>

						<div className="mt-8 flex flex-col justify-center gap-4 sm:flex-row">
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
			</div>
		</section>
	);
}
