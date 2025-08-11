import { ArrowLeft, Check, ChevronRight, Copy, Rocket } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { QuickstartExamples } from "@/components/ui/quickstart-examples";
import { Separator } from "@/components/ui/separator";

interface QuickstartStepProps {
	apiKey: string;
	onContinue: () => void;
	onBack: () => void;
}

export function QuickstartStep({
	apiKey,
	onContinue,
	onBack,
}: QuickstartStepProps) {
	const [copiedApiKey, setCopiedApiKey] = useState(false);

	const handleCopyApiKey = async () => {
		if (apiKey) {
			try {
				await navigator.clipboard.writeText(apiKey);
				setCopiedApiKey(true);
				setTimeout(() => setCopiedApiKey(false), 2000);
			} catch (err) {
				console.error("Failed to copy:", err);
			}
		}
	};

	return (
		<Card className="border-2">
			<CardHeader>
				<div className="mb-4 flex items-center gap-2">
					<Button variant="ghost" size="sm" onClick={onBack} className="p-2">
						<ArrowLeft className="h-4 w-4" />
					</Button>
					<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
						<Rocket className="h-5 w-5 text-primary" />
					</div>
				</div>
				<CardTitle className="text-xl">Your API key is ready!</CardTitle>
				<CardDescription>
					Save your API key and test it with these examples.
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-6">
				{/* API Key Display */}
				<div className="space-y-3">
					<div className="space-y-2">
						<p className="font-semibold text-muted-foreground text-sm">
							Please save this API key somewhere safe. You won't be able to view
							it again.
						</p>
					</div>
					<div className="space-y-2">
						<div className="flex items-center justify-between">
							<span className="font-medium text-sm">API Key</span>
							<Button
								variant="outline"
								size="sm"
								onClick={handleCopyApiKey}
								className="h-8 px-3 text-xs"
							>
								{copiedApiKey ? (
									<>
										<Check className="mr-1 h-3 w-3" />
										Copied
									</>
								) : (
									<>
										<Copy className="mr-1 h-3 w-3" />
										Copy
									</>
								)}
							</Button>
						</div>
						<div className="rounded-md border bg-muted p-3">
							<code className="break-all font-mono text-sm">{apiKey}</code>
						</div>
					</div>
				</div>

				<Separator />

				{/* Quick Examples */}
				<QuickstartExamples
					apiKey={apiKey}
					title="🚀 Test Your API"
					description="Try these examples to get started"
				/>

				<div className="flex justify-end gap-3">
					<Button type="button" variant="outline" onClick={onBack}>
						Back
					</Button>
					<Button onClick={onContinue} className="min-w-32">
						Continue to Dashboard
						<ChevronRight className="ml-2 h-4 w-4" />
					</Button>
				</div>
			</CardContent>
		</Card>
	);
}
