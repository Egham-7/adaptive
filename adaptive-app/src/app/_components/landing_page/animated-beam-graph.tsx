"use client";

import { motion } from "framer-motion";
import { MessageSquare, Zap } from "lucide-react";
import Image from "next/image";
import { forwardRef, useCallback, useRef, useState } from "react";
import { AnimatedBeam } from "@/components/ui/animated-beam";
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
				"z-10 flex size-20 items-center justify-center rounded-full bg-gradient-to-r from-primary to-primary/80 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]",
				className,
			)}
		>
			<MessageSquare className="h-8 w-8 text-primary-foreground" />
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

export default function AnimatedBeamGraph() {
	const containerRef = useRef<HTMLDivElement>(null);
	const promptRef = useRef<HTMLDivElement>(null);
	const adaptiveRef = useRef<HTMLDivElement>(null);
	const openaiRef = useRef<HTMLDivElement>(null);
	const anthropicRef = useRef<HTMLDivElement>(null);
	const googleRef = useRef<HTMLDivElement>(null);
	const metaRef = useRef<HTMLDivElement>(null);

	const [circlesReady, setCirclesReady] = useState<CircleState>({
		prompt: false,
		adaptive: false,
		openai: false,
		anthropic: false,
		google: false,
		meta: false,
	});

	const markCircleReady = useCallback((circleName: keyof CircleState) => {
		setCirclesReady((prev) => ({ ...prev, [circleName]: true }));
	}, []);

	const shouldRenderBeam1 = circlesReady.prompt && circlesReady.adaptive;
	const shouldRenderBeam2 = circlesReady.adaptive && circlesReady.openai;
	const shouldRenderBeam3 = circlesReady.adaptive && circlesReady.anthropic;
	const shouldRenderBeam4 = circlesReady.adaptive && circlesReady.google;
	const shouldRenderBeam5 = circlesReady.adaptive && circlesReady.meta;

	return (
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

			{/* Animated Beams */}
			{shouldRenderBeam1 && (
				<AnimatedBeam
					containerRef={containerRef as React.RefObject<HTMLElement>}
					fromRef={promptRef as React.RefObject<HTMLElement>}
					toRef={adaptiveRef as React.RefObject<HTMLElement>}
					delay={0.2}
					duration={2}
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
	);
}