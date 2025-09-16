"use client";

import { SignedOut, SignUpButton } from "@clerk/nextjs";
import { motion } from "framer-motion";
import {
	Building2,
	Code,
	FolderPlus,
	Key,
	Rocket,
	TrendingDown,
	UserPlus,
} from "lucide-react";
import { Button } from "@/components/ui/button";

export default function GetStartedSection() {
	const steps = [
		{
			number: 1,
			title: "Sign up for free",
			description:
				"Create your account and get $3.14 in free credits to start optimizing",
			icon: UserPlus,
		},
		{
			number: 2,
			title: "Create your organization",
			description:
				"Set up your workspace to manage your AI projects and team members",
			icon: Building2,
		},
		{
			number: 3,
			title: "Create your project",
			description:
				"Organize your API keys and track usage by project or application",
			icon: FolderPlus,
		},
		{
			number: 4,
			title: "Generate your API key",
			description:
				"Create secure API keys with custom permissions and rate limits",
			icon: Key,
		},
		{
			number: 5,
			title: "Use with quickstart examples",
			description: "Test your setup with cURL, JavaScript, or Python examples",
			icon: Code,
		},
		{
			number: 6,
			title: "See 60-90% savings",
			description:
				"Watch real-time cost reductions across OpenAI, Anthropic, Google, and any provider",
			icon: TrendingDown,
		},
	];

	const containerVariants = {
		hidden: { opacity: 0 },
		visible: {
			opacity: 1,
			transition: {
				staggerChildren: 0.2,
			},
		},
	};

	const itemVariants = {
		hidden: { opacity: 0, y: 50 },
		visible: { opacity: 1, y: 0 },
	};

	return (
		<SignedOut>
			<section className="overflow-hidden py-16 md:py-32">
				<div className="mx-auto max-w-7xl px-6">
					<motion.div
						initial={{ opacity: 0, y: 30 }}
						whileInView={{ opacity: 1, y: 0 }}
						viewport={{ once: true }}
						transition={{ duration: 0.6 }}
						className="text-center"
					>
						<h2 className="font-semibold text-4xl lg:text-5xl">
							Get Started in{" "}
							<span className="bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent">
								6 Easy Steps
							</span>
						</h2>
						<p className="mx-auto mt-6 max-w-3xl text-balance text-lg text-muted-foreground">
							From signup to 60-90% cost savings in minutes. No training data,
							no model profiling, no weeks of setup - just instant savings
							across any provider.
						</p>
						<p className="mx-auto mt-2 max-w-2xl font-medium text-sm text-warning">
							⏰ While competitors spend weeks updating routers, you're already
							using the latest models
						</p>
					</motion.div>

					<motion.div
						variants={containerVariants}
						initial="hidden"
						whileInView="visible"
						viewport={{ once: true }}
						className="mt-16 grid grid-cols-1 items-start gap-8 sm:grid-cols-2 lg:grid-cols-3"
					>
						{steps.map((step, index) => {
							const Icon = step.icon;

							return (
								<motion.div
									key={step.number}
									variants={itemVariants}
									transition={{
										duration: 0.6,
										ease: [0.4, 0, 0.2, 1],
										delay: index * 0.1,
									}}
									className="flex h-full flex-col space-y-6"
								>
									{/* Content */}
									<div className="flex-grow space-y-4">
										{/* Step Circle and Icon */}
										<div className="flex flex-col items-center gap-3">
											{/* Step Circle */}
											<div className="relative">
												<div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 ring-2 ring-primary/20">
													<span className="font-bold text-2xl text-primary">
														{step.number}
													</span>
												</div>
												{/* Animated circle border */}
												<div className="absolute inset-0 animate-pulse rounded-full bg-gradient-to-r from-primary to-primary/60 opacity-20" />
											</div>

											{/* Icon */}
											<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
												<Icon className="h-5 w-5 text-primary" />
											</div>
										</div>

										<div className="space-y-3 text-center">
											<h3 className="font-semibold text-foreground text-xl">
												{step.title}
											</h3>
											<p className="text-muted-foreground text-sm leading-relaxed">
												{step.description}
											</p>
										</div>
									</div>
								</motion.div>
							);
						})}
					</motion.div>

					{/* Call to action */}
					<motion.div
						initial={{ opacity: 0, y: 30 }}
						whileInView={{ opacity: 1, y: 0 }}
						viewport={{ once: true }}
						transition={{ duration: 0.6, delay: 0.2 }}
						className="mt-16 space-y-6 text-center"
					>
						<p className="text-muted-foreground">
							Ready for 60-90% savings across any provider?{" "}
							<span className="font-medium text-primary">
								Free trial • $3.14 credit • No credit card required
							</span>
						</p>

						<div className="flex justify-center">
							<SignUpButton signInForceRedirectUrl="/api-platform/organizations">
								<Button
									size="lg"
									className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
								>
									<Rocket className="relative mr-2 size-4" aria-hidden="true" />
									Start Saving 60-90%
								</Button>
							</SignUpButton>
						</div>
					</motion.div>
				</div>
			</section>
		</SignedOut>
	);
}
