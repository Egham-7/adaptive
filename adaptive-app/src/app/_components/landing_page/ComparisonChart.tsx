"use client";
import { useEffect, useRef, useState } from "react";
import {
	Bar,
	BarChart,
	CartesianGrid,
	Legend,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";

const baseData = [
	{
		name: "Adaptive",
		"Cost($)": 0.2,
		"Quality(%)": 95,
	},
	{
		name: "GPT-4o",
		"Cost($)": 12.5,
		"Quality(%)": 92,
	},
	{
		name: "Claude Sonnet 4",
		"Cost($)": 18,
		"Quality(%)": 90,
	},
];

export default function ComparisonChart() {
	const chartRef = useRef<HTMLDivElement>(null);
	const [targetProgress, setTargetProgress] = useState(0);
	const [animatedProgress, setAnimatedProgress] = useState(0);
	const [shouldAnimate, setShouldAnimate] = useState(false);

	// Get theme colors from CSS variables
	const [barColors, setBarColors] = useState({
		primary: "oklch(0.4341 0.0392 41.9938)",
		secondary: "oklch(0.65 0.0651 74.3695)", // Reduced lightness from 0.92 to 0.65 for better visibility
	});

	useEffect(() => {
		function updateColors() {
			setBarColors({
				primary:
					getComputedStyle(document.documentElement)
						.getPropertyValue("--color-primary")
						.trim() || "oklch(0.4341 0.0392 41.9938)",
				secondary:
					getComputedStyle(document.documentElement)
						.getPropertyValue("--color-secondary")
						.trim() || "oklch(0.65 0.0651 74.3695)", // Darker fallback
			});
		}
		updateColors();
		window.addEventListener("themechange", updateColors); // If you have a custom event for theme change
		return () => window.removeEventListener("themechange", updateColors);
	}, []);

	// Track scroll and set the target progress
	useEffect(() => {
		const handleScroll = () => {
			if (!chartRef.current) return;
			const rect = chartRef.current.getBoundingClientRect();
			const windowHeight =
				window.innerHeight || document.documentElement.clientHeight;
			const chartHeight = rect.height;
			const visible =
				Math.min(windowHeight, rect.bottom) - Math.max(0, rect.top);
			let newProgress = visible / chartHeight;
			newProgress = Math.max(0, Math.min(1, newProgress));
			setTargetProgress(newProgress);
		};
		window.addEventListener("scroll", handleScroll, { passive: true });
		window.addEventListener("resize", handleScroll);
		handleScroll();
		return () => {
			window.removeEventListener("scroll", handleScroll);
			window.removeEventListener("resize", handleScroll);
		};
	}, []);

	// Smoothly animate the progress toward the target
	useEffect(() => {
		setAnimatedProgress(targetProgress);
		// Trigger animation when chart becomes visible
		if (targetProgress > 0.3 && !shouldAnimate) {
			setShouldAnimate(true);
		}
	}, [targetProgress, shouldAnimate]);

	// Create animated data based on scroll progress
	const animatedData = baseData.map((item) => ({
		...item,
		"Cost($)": item["Cost($)"] * Math.max(0, Math.min(1, animatedProgress)),
		"Quality(%)":
			item["Quality(%)"] * Math.max(0, Math.min(1, animatedProgress)),
	}));

	return (
		<div
			ref={chartRef}
			className={`w-full transition-all duration-700 ${
				animatedProgress > 0
					? "translate-y-0 opacity-100"
					: "translate-y-8 opacity-0"
			}`}
		>
			<Card className="w-full p-6 shadow-xl">
				<h4 className="mb-4 font-semibold text-xl">
					Cost (Per 1M Token) & Quality Comparison
				</h4>
				<ResponsiveContainer width="100%" height={300}>
					<BarChart
						data={animatedData} // Use animated data instead of baseData
						layout="vertical"
						margin={{ top: 16, right: 16, left: 40, bottom: 32 }}
					>
						<CartesianGrid strokeDasharray="3 3" />
						<XAxis type="number" domain={[0, 20]} xAxisId="left" />
						<XAxis type="number" domain={[80, 100]} xAxisId="right" hide />
						<YAxis type="category" dataKey="name" />
						<Tooltip />
						<Legend />
						<Bar
							xAxisId="left"
							dataKey="Cost($)"
							fill={barColors.primary}
							radius={[0, 8, 8, 0]}
							barSize={32}
							isAnimationActive={false} // Disable built-in animation since we're using scroll-based animation
							animationDuration={0}
						/>
						<Bar
							xAxisId="right"
							dataKey="Quality(%)"
							fill={barColors.secondary}
							radius={[0, 8, 8, 0]}
							barSize={32}
							isAnimationActive={false} // Disable built-in animation since we're using scroll-based animation
							animationDuration={0}
						/>
					</BarChart>
				</ResponsiveContainer>
			</Card>
		</div>
	);
}
