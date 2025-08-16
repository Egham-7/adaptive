import { ArrowLeft, ExternalLink } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default async function CodeExamplesPage({
	params,
}: {
	params: Promise<{ orgId: string; projectId: string }>;
}) {
	const { orgId, projectId } = await params;

	const endpoints = [
		{
			id: "chat-completions",
			title: "Chat Completions",
			description:
				"Send messages and get AI responses with intelligent model selection",
			method: "POST",
			endpoint: "/api/v1/chat/completions",
			icon: "💬",
		},
		{
			id: "select-model",
			title: "Model Selection",
			description:
				"Get the optimal model and provider for your request without executing",
			method: "POST",
			endpoint: "/api/v1/select-model",
			icon: "🎯",
		},
	];

	return (
		<div className="mx-auto max-w-7xl px-2 py-2 sm:px-4">
			{/* Back Navigation */}
			<div className="mb-6">
				<Button variant="ghost" size="sm" asChild>
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/quickstart`}
					>
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Quick Start
					</Link>
				</Button>
			</div>

			{/* Header Section */}
			<div className="mb-8">
				<div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
					<div>
						<h1 className="font-semibold text-3xl text-foreground">
							Code Examples
						</h1>
						<p className="text-muted-foreground">
							Comprehensive examples for all Adaptive API endpoints
						</p>
					</div>
					<div className="flex items-center gap-2">
						<Badge variant="secondary">OpenAI Compatible</Badge>
						<Badge variant="outline">REST API</Badge>
					</div>
				</div>
			</div>

			{/* API Endpoints */}
			<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
				{endpoints.map((endpoint) => (
					<Link
						key={endpoint.id}
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/examples/${endpoint.id}`}
					>
						<Card className="h-full transition-all hover:scale-[1.02] hover:shadow-md">
							<CardHeader>
								<div className="flex items-center gap-3">
									<span className="text-3xl">{endpoint.icon}</span>
									<div>
										<CardTitle className="flex items-center gap-2">
											{endpoint.title}
											<Badge variant="outline" className="text-xs">
												{endpoint.method}
											</Badge>
										</CardTitle>
									</div>
								</div>
							</CardHeader>
							<CardContent>
								<CardDescription className="mb-4">
									{endpoint.description}
								</CardDescription>
								<div className="rounded bg-muted p-3">
									<code className="font-mono text-muted-foreground text-sm">
										{endpoint.endpoint}
									</code>
								</div>
								<div className="mt-4 flex items-center gap-2 text-primary text-sm">
									<span>View examples</span>
									<ExternalLink className="h-4 w-4" />
								</div>
							</CardContent>
						</Card>
					</Link>
				))}
			</div>

			{/* Footer */}
			<div className="mt-8 text-center">
				<Separator className="mb-6" />
				<div className="flex items-center justify-center gap-4 text-muted-foreground text-sm">
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/quickstart`}
						className="flex items-center gap-1 hover:text-foreground"
					>
						<ArrowLeft className="h-3 w-3" />
						Quick Start
					</Link>
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}`}
						className="flex items-center gap-1 hover:text-foreground"
					>
						<ExternalLink className="h-3 w-3" />
						Dashboard
					</Link>
				</div>
			</div>
		</div>
	);
}
