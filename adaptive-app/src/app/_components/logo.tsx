import Image from "next/image";
import { useTheme } from "next-themes";

export function Logo() {
	const { theme } = useTheme();

	return (
		<div className="flex items-center">
			<Image
				src={
					theme === "dark"
						? "/logos/adaptive-dark.svg"
						: "/logos/adaptive-light.svg"
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
