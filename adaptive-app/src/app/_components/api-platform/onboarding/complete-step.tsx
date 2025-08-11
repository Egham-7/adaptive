import { CheckCircle, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";

interface CompleteStepProps {
	onComplete: () => void;
}

export function CompleteStep({ onComplete }: CompleteStepProps) {
	return (
		<Card className="border-2">
			<CardHeader className="text-center">
				<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-success/10">
					<CheckCircle className="h-8 w-8 text-success" />
				</div>
				<CardTitle className="text-2xl">You're all set!</CardTitle>
				<CardDescription className="text-base">
					Your organization and project have been created successfully. You can
					now start using the Adaptive AI Platform.
				</CardDescription>
			</CardHeader>
			<CardContent>
				<div className="space-y-4">
					<div className="space-y-2 rounded-lg bg-muted/50 p-4">
						<h4 className="font-medium">What's next?</h4>
						<ul className="space-y-1 text-muted-foreground text-sm">
							<li>• Generate API keys for your project</li>
							<li>• Invite team members to your organization</li>
							<li>• Start making API calls and track usage</li>
							<li>• Monitor your project's analytics and costs</li>
						</ul>
					</div>
					<Button onClick={onComplete} className="w-full" size="lg">
						Go to Dashboard
						<ChevronRight className="ml-2 h-4 w-4" />
					</Button>
				</div>
			</CardContent>
		</Card>
	);
}
