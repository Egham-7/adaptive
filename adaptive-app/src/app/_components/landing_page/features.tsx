import { BarChart3, Code2, Cpu, Network, Timer } from "lucide-react";
import Image from "next/image";
import { Card, CardContent } from "@/components/ui/card";
import { PerformanceMetricsChart } from "./performance-metrics-chart";
import { ProviderDownloadChart } from "./provider-download-chart";

export default function FeaturesSection() {
	return (
		<section
			id="features"
			className="bg-muted/50 py-16 md:py-32 dark:bg-transparent"
		>
			<div className="mx-auto max-w-3xl px-6 lg:max-w-5xl">
				<header className="mb-12 text-center">
					<h2 className="font-display font-semibold text-4xl lg:text-5xl">
						Features
					</h2>
					<p className="mt-4 text-muted-foreground">
						Intelligent AI optimization platform designed for modern
						applications
					</p>
				</header>
				<div className="grid grid-cols-6 gap-3">
					<Card className="col-span-full flex overflow-hidden lg:col-span-2">
						<CardContent className="m-auto flex flex-col items-center justify-center pt-6 text-center">
							<Timer className="h-16 w-16 text-primary" aria-hidden="true" />
							<h3 className="mt-6 font-semibold text-3xl">
								Smart Request Handling
							</h3>
							<p className="mt-2 text-muted-foreground">
								Automatically chooses between full AI models for complex tasks
								or specialized tiny models for simple questions to save time and
								money.
							</p>
						</CardContent>
					</Card>
					<Card className="relative col-span-full overflow-hidden md:col-span-3 lg:col-span-2">
						<CardContent className="pt-6">
							<div className="before:-inset-2 relative mx-auto flex aspect-square size-32 items-center justify-center rounded-full border before:absolute before:rounded-full before:border dark:border-white/10 dark:before:border-white/5">
								<Network className="h-12 w-12 text-primary" />
							</div>
							<div className="relative z-10 mt-6 space-y-2 text-center">
								<h2 className="font-medium text-lg transition hover:text-primary dark:text-white">
									Reliable AI Service
								</h2>
								<p className="text-foreground">
									Automatic backup to alternative AI providers ensures your
									applications stay online with proven reliability systems.
								</p>
							</div>
						</CardContent>
					</Card>
					<Card className="relative col-span-full overflow-hidden md:col-span-3 lg:col-span-2">
						<CardContent className="pt-6">
							<div className="pt-6 lg:px-6">
								<ProviderDownloadChart />
							</div>
							<div className="relative z-10 mt-14 space-y-2 text-center">
								<h2 className="font-medium text-lg transition">
									Universal Provider Support
								</h2>
								<p className="text-foreground">
									Connect to OpenAI, Anthropic, Google, Meta, and more through a
									single unified API with seamless provider switching.
								</p>
							</div>
						</CardContent>
					</Card>
					<Card className="relative col-span-full overflow-hidden lg:col-span-3">
						<div className="grid pt-6 md:grid-cols-2">
							<div className="relative z-10 flex flex-col justify-between space-y-12 px-6 lg:space-y-6">
								<div className="before:-inset-2 relative flex aspect-square size-12 rounded-full border before:absolute before:rounded-full before:border dark:border-white/10 dark:before:border-white/5">
									<Cpu className="m-auto size-5" strokeWidth={1} />
								</div>
								<div className="space-y-2 px-2">
									<h2 className="font-medium text-foreground text-lg transition hover:text-primary dark:text-white">
										Intelligent Model Selection
									</h2>
									<p className="text-foreground">
										Advanced algorithms automatically choose the best AI model
										and optimize settings for quality and cost efficiency.
									</p>
								</div>
							</div>
							<div className="-mb-6 -mr-6 relative mt-6 h-fit rounded-tl-(--radius) border-t border-l p-6 py-6 md:ml-6">
								<div className="absolute top-2 left-3 flex gap-1">
									<span className="block size-2 rounded-full border dark:border-white/10 dark:bg-white/10" />
									<span className="block size-2 rounded-full border dark:border-white/10 dark:bg-white/10" />
									<span className="block size-2 rounded-full border dark:border-white/10 dark:bg-white/10" />
								</div>
								<PerformanceMetricsChart />
							</div>
						</div>
					</Card>
					<Card className="relative col-span-full overflow-hidden lg:col-span-3">
						<CardContent className="grid h-full pt-6 md:grid-cols-2">
							<div className="relative z-10 flex flex-col justify-between space-y-12 lg:space-y-6">
								<div className="before:-inset-2 relative flex aspect-square size-12 rounded-full border before:absolute before:rounded-full before:border dark:border-white/10 dark:before:border-white/5">
									<BarChart3 className="m-auto size-6" strokeWidth={1} />
								</div>
								<div className="space-y-2">
									<h2 className="font-medium text-lg transition">
										Continuous Learning
									</h2>
									<p className="text-foreground">
										Our system learns from performance data to continuously
										improve model selection and deliver better results over
										time.
									</p>
								</div>
							</div>
							<div className="md:-my-6 md:-mr-6 relative mt-6 before:absolute before:inset-0 before:mx-auto before:w-px before:bg-(--color-border)">
								<div className="relative flex h-full flex-col justify-center space-y-6 py-6">
									<div className="relative flex w-[calc(50%+0.875rem)] items-center justify-end gap-2">
										<span className="block h-fit rounded border px-2 py-1 text-xs shadow-sm">
											Likeur
										</span>
										<div className="size-7 ring-4 ring-background">
											<Image
												className="size-full rounded-full"
												src="https://avatars.githubusercontent.com/u/102558960?v=4"
												alt="User avatar"
												width={28}
												height={28}
											/>
										</div>
									</div>
									<div className="relative ml-[calc(50%-1rem)] flex items-center gap-2">
										<div className="size-8 ring-4 ring-background">
											<Image
												className="size-full rounded-full"
												src="https://avatars.githubusercontent.com/u/47919550?v=4"
												alt="User avatar"
												width={32}
												height={32}
											/>
										</div>
										<span className="block h-fit rounded border px-2 py-1 text-xs shadow-sm">
											M. Irung
										</span>
									</div>
									<div className="relative flex w-[calc(50%+0.875rem)] items-center justify-end gap-2">
										<span className="block h-fit rounded border px-2 py-1 text-xs shadow-sm">
											B. Ng
										</span>
										<div className="size-7 ring-4 ring-background">
											<Image
												className="size-full rounded-full"
												src="https://avatars.githubusercontent.com/u/31113941?v=4"
												alt="User avatar"
												width={28}
												height={28}
											/>
										</div>
									</div>
								</div>
							</div>
						</CardContent>
					</Card>
					<Card className="relative col-span-full overflow-hidden lg:col-span-2">
						<CardContent className="pt-6">
							<div className="before:-inset-2 relative mx-auto flex aspect-square size-32 items-center justify-center rounded-full border before:absolute before:rounded-full before:border dark:border-white/10 dark:before:border-white/5">
								<Code2 className="h-12 w-12 text-primary" />
							</div>
							<div className="relative z-10 mt-6 space-y-2 text-center">
								<h2 className="font-medium text-lg transition hover:text-primary dark:text-white">
									Open Source & Transparent
								</h2>
								<p className="text-foreground">
									Built with transparency in mind. Our routing algorithms and
									infrastructure are open source, giving you full visibility and
									control.
								</p>
							</div>
						</CardContent>
					</Card>
				</div>
			</div>
		</section>
	);
}
