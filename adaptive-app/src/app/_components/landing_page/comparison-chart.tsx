"use client";
import { motion, useInView, useSpring, useTransform } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import {
	Bar,
	BarChart,
	CartesianGrid,
	Cell,
	Legend,
	XAxis,
	YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import { designSystemColors, getComparisonChartColor } from "@/lib/colors";

interface ChartData {
	name: string;
	costPerMillionTokens: number;
}

const baseData: ChartData[] = [
	{
		name: "Adaptive",
		costPerMillionTokens: 2.5, // $2.50 per million tokens (intelligent routing)
	},
	{
		name: "OpenAI",
		costPerMillionTokens: 15.0, // $15.00 per million tokens
	},
	{
		name: "Anthropic",
		costPerMillionTokens: 18.0, // $18.00 per million tokens
	},
	{
		name: "Gemini",
		costPerMillionTokens: 12.5, // $12.50 per million tokens
	},
];

const chartConfig = {
	costPerMillionTokens: {
		label: "Cost per Million Tokens ($)",
		color: designSystemColors.primary,
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
			costPerMillionTokens: 0,
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
						costPerMillionTokens: baseItem
							? baseItem.costPerMillionTokens * latest
							: 0,
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
					className="mb-4 font-semibold text-foreground text-xl"
					id="chart-title"
				>
					Cost per Million Tokens Comparison
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
							margin={{ top: 16, right: 16, left: 120, bottom: 60 }}
						>
							<CartesianGrid
								strokeDasharray="3 3"
								horizontal={true}
								vertical={false}
								stroke={designSystemColors.border}
								className="opacity-30"
							/>
							<XAxis
								type="number"
								domain={[0, 20]}
								tick={{
									fill: designSystemColors.mutedForeground,
									fontSize: 12,
								}}
								label={{
									value: "Cost per Million Tokens ($)",
									position: "insideBottom",
									offset: -5,
									style: {
										textAnchor: "middle",
										fill: designSystemColors.mutedForeground,
									},
								}}
							/>
							<YAxis
								type="category"
								dataKey="name"
								width={120}
								fontSize={12}
								tick={{
									fill: designSystemColors.mutedForeground,
									fontSize: 11,
								}}
								label={{
									value: "AI Provider",
									angle: -90,
									position: "insideLeft",
									style: {
										textAnchor: "middle",
										fill: designSystemColors.mutedForeground,
									},
								}}
							/>
							<ChartTooltip content={<ChartTooltipContent />} />
							<Legend
								verticalAlign="bottom"
								height={36}
								wrapperStyle={{ paddingTop: "20px" }}
							/>
							<Bar
								dataKey="costPerMillionTokens"
								radius={[0, 8, 8, 0]}
								barSize={30}
								isAnimationActive={false}
								animationDuration={0}
							>
								{animatedData.map((entry) => (
									<Cell
										key={`cost-cell-${entry.name}`}
										fill={getComparisonChartColor(entry.name === "Adaptive")}
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
