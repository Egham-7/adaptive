"use client";

import Image from "next/image";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function Logo() {
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
						? "/logos/adaptive-dark.svg"
						: "/logos/adaptive-light.png"
				}
				alt="Adaptive Logo"
				width={120}
				height={100}
				className="mr-2"
			/>
			<span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text font-bold font-display text-transparent text-xl">
				Adaptive
			</span>
		</div>
	);
}
