import { BackgroundBeams } from "./background-beams";
import { Globe } from "./globe";
export function WaitingListSection() {
	return (
		<div className="flexmax-w-lg relative items-center justify-center overflow-hidden rounded-lg border bg-background md:pb-60 md:shadow-xl">
			<div className="relative flex h-[20rem] w-full flex-col items-center justify-center rounded-md bg-background antialiased">
				<Globe />
				<div className="relative z-10 mx-auto max-w-2xl rounded-lg bg-gradient-to-br from-white/90 via-white/80 to-white/70 p-6 shadow-lg backdrop-blur-md dark:from-white/95 dark:via-white/90 dark:to-white/85">
					<h1 className="relative z-10 bg-gradient-to-b from-gray-900 to-gray-600 bg-clip-text text-center font-bold font-sans text-lg text-transparent md:text-7xl dark:from-gray-900 dark:to-gray-700">
						Join the waitlist
					</h1>

					<p className="relative z-10 mx-auto my-2 max-w-lg text-center text-gray-700 text-sm dark:text-gray-800">
						Help us launch this software as soon as possible!
					</p>
					<Input
						type="email"
						placeholder="hello@llmadaptive.uk"
						className="relative z-10 mt-4 w-full bg-white dark:bg-white"
					/>
				</div>
				<BackgroundBeams />
			</div>
		</div>
	);
}

import { Input } from "@/components/ui/input";
