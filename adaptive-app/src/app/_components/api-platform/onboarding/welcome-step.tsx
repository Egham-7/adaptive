import { Building2, ChevronRight, FolderPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";

interface WelcomeStepProps {
	onContinue: () => void;
}

export function WelcomeStep({ onContinue }: WelcomeStepProps) {
	return (
		<Card className="border-2">
			<CardHeader className="text-center">
				<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-primary/10">
					<Building2 className="h-8 w-8 text-primary" />
				</div>
				<CardTitle className="text-2xl">Let's start building</CardTitle>
				<CardDescription className="text-base">
					Organizations help you manage your AI projects and collaborate with
					your team. You can always create more organizations later.
				</CardDescription>
			</CardHeader>
			<CardContent>
				<div className="space-y-4">
					<div className="grid grid-cols-1 gap-4 md:grid-cols-2">
						<div className="flex items-start gap-3 rounded-lg bg-muted/50 p-4">
							<Building2 className="mt-1 h-6 w-6 text-primary" />
							<div>
								<h4 className="font-medium">Organization</h4>
								<p className="text-muted-foreground text-sm">
									Your workspace for managing projects and teams
								</p>
							</div>
						</div>
						<div className="flex items-start gap-3 rounded-lg bg-muted/50 p-4">
							<FolderPlus className="mt-1 h-6 w-6 text-primary" />
							<div>
								<h4 className="font-medium">Project</h4>
								<p className="text-muted-foreground text-sm">
									Individual AI projects with their own API keys and analytics
								</p>
							</div>
						</div>
					</div>
					<Button onClick={onContinue} className="w-full" size="lg">
						Get Started
						<ChevronRight className="ml-2 h-4 w-4" />
					</Button>
				</div>
			</CardContent>
		</Card>
	);
}
