interface PromptSuggestionsProps {
	label: string;
	sendMessage: (message: { text: string }) => void;
	suggestions: string[];
}

export function PromptSuggestions({
	label,
	sendMessage,
	suggestions,
}: PromptSuggestionsProps) {
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
