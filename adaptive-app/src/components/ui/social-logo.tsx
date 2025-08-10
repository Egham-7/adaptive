"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useTheme } from "next-themes";

interface SocialLogoProps {
	width?: number;
	height?: number;
	className?: string;
	alt?: string;
}

export function SocialLogo({ 
	width = 120, 
	height = 120, 
	className = "", 
	alt = "Adaptive Social Logo" 
}: SocialLogoProps) {
	const { resolvedTheme } = useTheme();
	const [mounted, setMounted] = useState(false);

	useEffect(() => {
		setMounted(true);
	}, []);

	if (!mounted) {
		return null;
	}

	return (
		<Image
			src={
				resolvedTheme === "dark"
					? "/logos/adaptive-social-dark.svg"
					: "/logos/adaptive-social-light.svg"
			}
			alt={alt}
			width={width}
			height={height}
			className={className}
		/>
	);
}