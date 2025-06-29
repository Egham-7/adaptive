import { TypingLoader } from "./loader";

export function TypingIndicator() {
	return (
		<div className="justify-left flex space-x-1">
			<div className="rounded-lg bg-muted p-3">
				<TypingLoader size="md" />
			</div>
		</div>
	);
}
