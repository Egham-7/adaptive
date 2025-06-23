"use client";
import { useRef, useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card } from "@/components/ui/card";

const baseData = [
  {
    name: "Adaptive",
    Cost: 1,
    Quality: 95,
  },
  {
    name: "GPT-4",
    Cost: 4,
    Quality: 92,
  },
  {
    name: "Claude 3",
    Cost: 3.5,
    Quality: 90,
  },
];

export default function ComparisonChart() {
  const chartRef = useRef<HTMLDivElement>(null);
  const [targetProgress, setTargetProgress] = useState(0);
  const [animatedProgress, setAnimatedProgress] = useState(0);

  // Get theme colors from CSS variables
  const [barColors, setBarColors] = useState({
    primary: "#6366f1",
    secondary: "#22d3ee",
  });

  useEffect(() => {
    function updateColors() {
      setBarColors({
        primary:
          getComputedStyle(document.documentElement)
            .getPropertyValue("--color-primary")
            .trim() || "#6366f1",
        secondary:
          getComputedStyle(document.documentElement)
            .getPropertyValue("--color-secondary")
            .trim() || "#22d3ee",
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
  }, [targetProgress]);

  const animatedData = baseData.map((d) => ({
    ...d,
    Cost: d.Cost * animatedProgress,
    Quality: d.Quality * animatedProgress,
  }));

  return (
    <div
      ref={chartRef}
      className={`transition-all duration-700 w-full ${
        animatedProgress > 0
          ? "opacity-100 translate-y-0"
          : "opacity-0 translate-y-8"
      }`}
    >
      <Card className="p-6 w-full shadow-xl">
        <h4 className="font-semibold text-xl mb-4">
          Cost & Quality Comparison
        </h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={animatedData}
            layout="vertical"
            margin={{ top: 16, right: 16, left: 40, bottom: 32 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 5]} xAxisId="left" />
            <XAxis type="number" domain={[80, 100]} xAxisId="right" hide />
            <YAxis type="category" dataKey="name" />
            <Tooltip />
            <Legend />
            <Bar
              xAxisId="left"
              dataKey="Cost"
              fill={barColors.primary}
              radius={[0, 8, 8, 0]}
              barSize={32}
              isAnimationActive={false}
            />
            <Bar
              xAxisId="right"
              dataKey="Quality"
              fill={barColors.secondary}
              radius={[0, 8, 8, 0]}
              barSize={32}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
