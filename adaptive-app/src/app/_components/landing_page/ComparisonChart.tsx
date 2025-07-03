"use client";
import { motion, useInView, useSpring, useTransform } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from "recharts";
import { Card } from "@/components/ui/card";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";

interface ChartData {
	name: string;
	costSavedPercent: number;
}

const baseData: ChartData[] = [
	{
		name: "Adaptive",
		costSavedPercent: 78, // 78% cost savings per million tokens
	},
	{
		name: "OpenAI",
		costSavedPercent: 35, // 35% cost savings with optimizations
	},
	{
		name: "Anthropic",
		costSavedPercent: 22, // 22% cost savings with optimizations
	},
	{
		name: "Google",
		costSavedPercent: 15, // 15% cost savings with optimizations
	},
];

const chartConfig = {
	costSavedPercent: {
		label: "Cost Saved %",
		color: "#6366f1", // Primary color mapped from CSS vars
	},
} satisfies ChartConfig;

export default function ComparisonChart() {
	const chartRef = useRef<HTMLDivElement>(null);
	const isInView = useInView(chartRef, {
		once: true,
		margin: "-100px",
		amount: 0.3,
	});

	// Smooth spring animation for progress
	const springProgress = useSpring(0, {
		stiffness: 100,
		damping: 30,
		restDelta: 0.001,
	});

	// Transform progress to animated values
	const animatedCost = useTransform(springProgress, [0, 1], [0, 1]);

	// State for animated data
	const [animatedData, setAnimatedData] = useState<ChartData[]>(
		baseData.map((item) => ({
			...item,
			costSavedPercent: 0,
		})),
	);

	// Trigger animation when in view
	useEffect(() => {
		if (isInView) {
			springProgress.set(1);
		}
	}, [isInView, springProgress]);

	// Update animated data based on spring values
	useEffect(() => {
		const unsubscribeCost = animatedCost.on("change", (latest) => {
			setAnimatedData((prev) =>
				prev.map((item, index) => {
					const baseItem = baseData[index];
					return {
						...item,
						costSavedPercent: baseItem ? baseItem.costSavedPercent * latest : 0,
					};
				}),
			);
		});

		return () => {
			unsubscribeCost();
		};
	}, [animatedCost]);

	return (
		<motion.div
			ref={chartRef}
			initial={{ opacity: 0, y: 50 }}
			animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
			transition={{
				duration: 0.8,
				ease: [0.21, 1.11, 0.81, 0.99], // Custom ease for bounce effect
			}}
			className="w-full"
		>
			<Card
				className="w-full overflow-hidden p-6 shadow-xl"
				aria-labelledby="chart-title"
			>
				<motion.h4
					initial={{ opacity: 0, x: -20 }}
					animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
					transition={{ delay: 0.2, duration: 0.6 }}
					className="mb-4 font-semibold text-xl"
					id="chart-title"
					style={{ color: "#0f172a" }}
				>
					Provider Cost Efficiency Comparison
				</motion.h4>

				<motion.div
					initial={{ opacity: 0, scale: 0.9 }}
					animate={
						isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }
					}
					transition={{
						delay: 0.4,
						duration: 0.7,
						ease: [0.25, 0.46, 0.45, 0.94],
					}}
				>
					<ChartContainer config={chartConfig} className="h-[400px] w-full">
						<BarChart
							data={animatedData}
							layout="vertical"
							margin={{ top: 16, right: 16, left: 120, bottom: 32 }}
						>
							<CartesianGrid
								strokeDasharray="3 3"
								horizontal={true}
								vertical={false}
								stroke="#e2e8f0"
								className="opacity-30"
							/>
							<XAxis
								type="number"
								domain={[0, 100]}
								tickFormatter={(value) => `${value}%`}
								tick={{ fill: "#64748b", fontSize: 12 }}
							/>
							<YAxis
								type="category"
								dataKey="name"
								width={120}
								fontSize={12}
								tick={{ fill: "#64748b", fontSize: 11 }}
							/>
							<ChartTooltip
								content={
									<ChartTooltipContent
										formatter={(value, _name) => [
											`${Number(value).toFixed(0)}%`,
										]}
									/>
								}
							/>
							<Bar
								dataKey="costSavedPercent"
								radius={[0, 8, 8, 0]}
								barSize={30}
								isAnimationActive={false}
								animationDuration={0}
							>
								{animatedData.map((entry) => (
									<Cell
										key={`cost-cell-${entry.name}`}
										fill={
											entry.name === "Adaptive"
												? "#059669" // Green for Adaptive (savings)
												: "#64748b" // Muted gray for others
										}
									/>
								))}
							</Bar>
						</BarChart>
					</ChartContainer>
				</motion.div>
			</Card>
		</motion.div>
	);
}
