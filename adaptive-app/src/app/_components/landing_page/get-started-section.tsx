"use client";

import { SignedOut, SignUpButton } from "@clerk/nextjs";
import { motion } from "framer-motion";
import { Rocket } from "lucide-react";
import { Button } from "@/components/ui/button";
import VideoPlayer from "@/components/ui/video-player";

export default function GetStartedSection() {

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
							See How{" "}
							<span className="bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent">
								Adaptive Works
							</span>
						</h2>
						<p className="mx-auto mt-6 max-w-3xl text-balance text-lg text-muted-foreground">
							Watch how Adaptive reduces your AI costs by 60-90% with intelligent
							model routing and automatic optimization - no setup required.
						</p>
					</motion.div>

					<motion.div
						initial={{ opacity: 0, y: 30 }}
						whileInView={{ opacity: 1, y: 0 }}
						viewport={{ once: true }}
						transition={{ duration: 0.6, delay: 0.2 }}
						className="mt-16"
					>
						<VideoPlayer src="/demo-video.mp4" />
					</motion.div>

					{/* Call to action */}
					<motion.div
						initial={{ opacity: 0, y: 30 }}
						whileInView={{ opacity: 1, y: 0 }}
						viewport={{ once: true }}
						transition={{ duration: 0.6, delay: 0.4 }}
						className="mt-16 space-y-6 text-center"
					>
						<p className="text-muted-foreground">
							Ready to start saving 60-90% on your AI costs?{" "}
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
									Start Saving Now
								</Button>
							</SignUpButton>
						</div>
					</motion.div>
				</div>
			</section>
		</SignedOut>
	);
}
