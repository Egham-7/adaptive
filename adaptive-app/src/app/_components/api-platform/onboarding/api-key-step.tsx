import { ArrowLeft, ChevronRight, Key } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";

interface ApiKeyStepProps {
	onCreateApiKey: () => void;
	onSkip: () => void;
	onBack: () => void;
	isLoading: boolean;
}

export function ApiKeyStep({
	onCreateApiKey,
	onSkip,
	onBack,
	isLoading,
}: ApiKeyStepProps) {
	return (
		<Card className="border-2">
			<CardHeader>
				<div className="mb-4 flex items-center gap-2">
					<Button variant="ghost" size="sm" onClick={onBack} className="p-2">
						<ArrowLeft className="h-4 w-4" />
					</Button>
					<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
						<Key className="h-5 w-5 text-primary" />
					</div>
				</div>
				<CardTitle className="text-xl">
					Generate your API key (Optional)
				</CardTitle>
				<CardDescription>
					Create your first API key to start making requests to the Adaptive AI
					Platform.
				</CardDescription>
			</CardHeader>
			<CardContent>
				<div className="space-y-6">
					<div className="rounded-lg bg-muted/50 p-4">
						<h4 className="mb-2 font-medium">What you'll get:</h4>
						<ul className="space-y-1 text-muted-foreground text-sm">
							<li>• A secure API key for your project</li>
							<li>• OpenAI-compatible endpoint access</li>
							<li>• Usage tracking and analytics</li>
							<li>• Cost monitoring and optimization</li>
						</ul>
					</div>
					<div className="flex justify-between">
						<Button type="button" variant="outline" onClick={onBack}>
							Back
						</Button>
						<div className="flex gap-3">
							<Button type="button" variant="ghost" onClick={onSkip}>
								Skip for now
							</Button>
							<Button
								onClick={onCreateApiKey}
								disabled={isLoading}
								className="min-w-32"
							>
								{isLoading ? "Creating..." : "Generate API Key"}
								<ChevronRight className="ml-2 h-4 w-4" />
							</Button>
						</div>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
