export function Logo() {
	return (
		<div className="flex items-center">
			<svg
				width="32"
				height="32"
				viewBox="0 0 32 32"
				fill="none"
				xmlns="http://www.w3.org/2000/svg"
				className="mr-2"
			>
				<title>Adaptive App Logo</title>
				<path
					d="M8 8L16 0L24 8V24L16 32L8 24V8Z"
					className="fill-primary-600 dark:fill-primary-500"
				/>
				<path
					d="M16 6L22 12V20L16 26L10 20V12L16 6Z"
					className="fill-white dark:fill-zinc-900"
				/>
			</svg>
			<span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text font-bold font-display text-transparent text-xl">
				Adaptive
			</span>
		</div>
	);
}
