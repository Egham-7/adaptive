import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, Code, PenTool } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface PromptSuggestionsProps {
	label: string;
	sendMessage: (message: { text: string }) => void;
	suggestions: string[];
	enableCategories?: boolean;
}

export function PromptSuggestions({
	label,
	sendMessage,
	suggestions,
	enableCategories = false,
}: PromptSuggestionsProps) {
	const [activeCommandCategory, setActiveCommandCategory] = useState<string | null>(null);

	const commandSuggestions = {
		learn: [
			"Explain the Big Bang theory",
			"How does photosynthesis work?",
			"What are black holes?",
			"Explain quantum computing",
			"How does the human brain work?",
		],
		code: [
			"Create a React component for a todo list",
			"Write a Python function to sort a list",
			"How to implement authentication in Next.js",
			"Explain async/await in JavaScript",
			"Create a CSS animation for a button",
		],
		write: [
			"Write a professional email to a client",
			"Create a product description for a smartphone",
			"Draft a blog post about AI",
			"Write a creative story about space exploration",
			"Create a social media post about sustainability",
		],
	};

	const handleCommandSelect = (command: string) => {
		sendMessage({ text: command });
		setActiveCommandCategory(null);
	};

	if (enableCategories) {
		return (
			<div className="w-full space-y-6">
				<div className="text-center">
					<motion.div
						initial={{ opacity: 0, y: 10 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.3 }}
						className="flex flex-col items-center"
					>
						<h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60 mb-2">
							Ready to assist you
						</h1>
						<p className="text-muted-foreground max-w-md">
							Ask me anything or try one of the suggestions below
						</p>
					</motion.div>
				</div>

				{/* Command categories */}
				<div className="w-full grid grid-cols-3 gap-4 mb-4">
					<CommandButton
						icon={<BookOpen className="w-5 h-5" />}
						label="Learn"
						isActive={activeCommandCategory === "learn"}
						onClick={() =>
							setActiveCommandCategory(
								activeCommandCategory === "learn" ? null : "learn"
							)
						}
					/>
					<CommandButton
						icon={<Code className="w-5 h-5" />}
						label="Code"
						isActive={activeCommandCategory === "code"}
						onClick={() =>
							setActiveCommandCategory(
								activeCommandCategory === "code" ? null : "code"
							)
						}
					/>
					<CommandButton
						icon={<PenTool className="w-5 h-5" />}
						label="Write"
						isActive={activeCommandCategory === "write"}
						onClick={() =>
							setActiveCommandCategory(
								activeCommandCategory === "write" ? null : "write"
							)
						}
					/>
				</div>

				{/* Command suggestions */}
				<AnimatePresence>
					{activeCommandCategory && (
						<motion.div
							initial={{ opacity: 0, height: 0 }}
							animate={{ opacity: 1, height: "auto" }}
							exit={{ opacity: 0, height: 0 }}
							className="w-full overflow-hidden"
						>
							<div className="bg-background rounded-xl border border-border shadow-sm overflow-hidden">
								<div className="p-3 border-b border-border">
									<h3 className="text-sm font-medium text-foreground">
										{activeCommandCategory === "learn"
											? "Learning suggestions"
											: activeCommandCategory === "code"
											? "Coding suggestions"
											: "Writing suggestions"}
									</h3>
								</div>
								<ul className="divide-y divide-border">
									{commandSuggestions[
										activeCommandCategory as keyof typeof commandSuggestions
									].map((suggestion, index) => (
										<motion.li
											key={index}
											initial={{ opacity: 0 }}
											animate={{ opacity: 1 }}
											transition={{ delay: index * 0.03 }}
											onClick={() => handleCommandSelect(suggestion)}
											className="p-3 hover:bg-muted cursor-pointer transition-colors duration-75"
										>
											<div className="flex items-center gap-3">
												{activeCommandCategory === "learn" ? (
													<BookOpen className="w-4 h-4 text-primary" />
												) : activeCommandCategory === "code" ? (
													<Code className="w-4 h-4 text-primary" />
												) : (
													<PenTool className="w-4 h-4 text-primary" />
												)}
												<span className="text-sm text-foreground">
													{suggestion}
												</span>
											</div>
										</motion.li>
									))}
								</ul>
							</div>
						</motion.div>
					)}
				</AnimatePresence>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			<h2 className="text-center font-bold text-2xl">{label}</h2>
			<div className="flex gap-6 text-sm">
				{suggestions.map((suggestion) => (
					<button
						type="button"
						key={suggestion}
						onClick={() => sendMessage({ text: suggestion })}
						className="h-max flex-1 rounded-xl border bg-background p-4 hover:bg-muted"
					>
						<p>{suggestion}</p>
					</button>
				))}
			</div>
		</div>
	);
}

interface CommandButtonProps {
	icon: React.ReactNode;
	label: string;
	isActive: boolean;
	onClick: () => void;
}

function CommandButton({ icon, label, isActive, onClick }: CommandButtonProps) {
	return (
		<motion.button
			onClick={onClick}
			className={cn(
				"flex flex-col items-center justify-center gap-2 p-4 rounded-xl border transition-all",
				isActive
					? "bg-primary/10 border-primary/20 shadow-sm"
					: "bg-background border-border hover:border-border/60"
			)}
		>
			<div className={cn(isActive ? "text-primary" : "text-muted-foreground")}>
				{icon}
			</div>
			<span
				className={cn(
					"text-sm font-medium",
					isActive ? "text-primary" : "text-foreground"
				)}
			>
				{label}
			</span>
		</motion.button>
	);
}
