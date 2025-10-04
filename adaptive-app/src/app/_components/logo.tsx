"use client";

import Image from "next/image";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

interface LogoProps {
	imageWidth?: number;
	imageHeight?: number;
	textSize?: "sm" | "base" | "lg" | "xl" | "2xl";
	showText?: boolean;
}

const textSizeMap = {
	sm: "text-sm",
	base: "text-base",
	lg: "text-lg",
	xl: "text-xl",
	"2xl": "text-2xl",
};

export function Logo({
	imageWidth = 120,
	imageHeight = 100,
	textSize = "xl",
	showText = true,
}: LogoProps) {
	const { resolvedTheme } = useTheme();
	const [mounted, setMounted] = useState(false);

	useEffect(() => {
		setMounted(true);
	}, []);

	if (!mounted) {
		return null;
	}

	return (
		<div className="flex items-center">
			<Image
				src={
					resolvedTheme === "dark"
						? "/logos/adaptive-dark.png"
						: "/logos/adaptive-light.png"
				}
				alt="Adaptive Logo"
				width={imageWidth}
				height={imageHeight}
				className="mr-2"
			/>
			{showText && (
				<span
					className={`bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text font-bold font-display text-transparent ${textSizeMap[textSize]}`}
				>
					Adaptive
				</span>
			)}
		</div>
	);
}
