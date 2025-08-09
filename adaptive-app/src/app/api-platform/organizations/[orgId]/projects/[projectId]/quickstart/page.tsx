"use client";

import {
	AlertCircle,
	ArrowLeft,
	Check,
	Copy,
	ExternalLink,
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/trpc/react";

const EXAMPLE_API_KEY = "your_api_key_here_placeholder_12345";
const API_BASE_URL =
	process.env.NEXT_PUBLIC_URL ??
	(typeof window !== "undefined" ? window.location.origin : "");

export default function QuickstartPage() {
	const { orgId, projectId } = useParams<{
		orgId: string;
		projectId: string;
	}>();

	// Fetch project API keys for examples
	const {
		data: apiKeys,
		isLoading: apiKeysLoading,
		isError: apiKeysError,
	} = api.api_keys.getByProject.useQuery({ projectId });

	const firstApiKey = apiKeys?.[0];
	const exampleKey = firstApiKey ? firstApiKey.key_preview : EXAMPLE_API_KEY;

	const copyToClipboard = async (text: string, label: string) => {
		try {
			await navigator.clipboard.writeText(text);
			toast.success(`${label} copied to clipboard!`);
		} catch {
			toast.error(`Failed to copy ${label}.`);
		}
	};

	const CodeBlock = ({
		code,
		language,
		title,
	}: {
		code: string;
		language: string;
		title?: string;
	}) => {
		const [copied, setCopied] = useState(false);

		const handleCopy = async () => {
			await copyToClipboard(code, title || "Code");
			setCopied(true);
			setTimeout(() => setCopied(false), 2000);
		};

		return (
			<div className="relative overflow-hidden rounded-lg border bg-muted/50">
				{title && (
					<div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2">
						<span className="font-medium text-sm">{title}</span>
						<Badge variant="secondary" className="text-xs">
							{language}
						</Badge>
					</div>
				)}
				<div className="relative">
					<pre className="max-w-full overflow-x-auto p-4 text-sm">
						<code className="whitespace-pre-wrap break-all sm:whitespace-pre sm:break-normal">
							{code}
						</code>
					</pre>
					<Button
						variant="outline"
						size="sm"
						onClick={handleCopy}
						className="absolute top-2 right-2"
					>
						{copied ? (
							<Check className="h-4 w-4" />
						) : (
							<Copy className="h-4 w-4" />
						)}
					</Button>
				</div>
			</div>
		);
	};

	const curlExample = `curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}..." \\
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user", 
        "content": "Hello! How are you today?"
      }
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'`;

	const jsExample = `import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: '${exampleKey}...',
  baseURL: '${API_BASE_URL}/api/v1',
});

async function main() {
  const completion = await client.chat.completions.create({
    messages: [
      { 
        role: 'user', 
        content: 'Hello! How are you today?' 
      }
    ],
    model: 'gpt-4o',
    max_tokens: 150,
    temperature: 0.7,
  });

  console.log(completion.choices[0]);
}

main();`;

	const pythonExample = `from openai import OpenAI

client = OpenAI(
    api_key="${exampleKey}...",
    base_url="${API_BASE_URL}/api/v1"
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Hello! How are you today?"
        }
    ],
    max_tokens=150,
    temperature=0.7
)

print(completion.choices[0].message.content)`;

	const exampleResponse = `{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. I'm here and ready to help you with any questions or tasks you might have. How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 31,
    "total_tokens": 40
  }
}`;

	return (
		<div className="mx-auto max-w-7xl px-2 py-2 sm:px-4">
			{/* Back Navigation */}
			<div className="mb-6">
				<Button variant="ghost" size="sm" asChild>
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}`}
					>
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Dashboard
					</Link>
				</Button>
			</div>

			{/* Header Section */}
			<div className="mb-8">
				<div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
					<div>
						<h1 className="font-semibold text-3xl text-foreground">
							Quick Start Guide
						</h1>
						<p className="text-muted-foreground">
							Get up and running with the Adaptive API in minutes
						</p>
					</div>
					<div className="flex items-center gap-2">
						<Badge variant="secondary">OpenAI Compatible</Badge>
					</div>
				</div>
			</div>

			{/* Main Content */}
			<div className="space-y-8">
				{/* Step 1: Get API Key */}
				<Card>
					<CardHeader>
						<div className="flex items-center gap-3">
							<div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary font-bold text-primary-foreground text-sm">
								1
							</div>
							<div>
								<CardTitle>Get Your API Key</CardTitle>
								<CardDescription>
									First, you'll need an API key to authenticate your requests
								</CardDescription>
							</div>
						</div>
					</CardHeader>
					<CardContent className="space-y-4">
						{apiKeysLoading ? (
							<div className="rounded-lg border p-4">
								<div className="flex items-center gap-2">
									<Skeleton className="h-4 w-4 rounded" />
									<Skeleton className="h-4 w-32" />
								</div>
								<Skeleton className="mt-2 h-3 w-full max-w-xs" />
							</div>
						) : apiKeysError ? (
							<div className="rounded-lg border bg-red-50 p-4 dark:bg-red-950/20">
								<div className="flex items-center gap-2 text-red-700 dark:text-red-400">
									<AlertCircle className="h-4 w-4" />
									<span className="font-medium text-sm">
										Error Loading API Keys
									</span>
								</div>
								<p className="mt-2 text-red-600 text-sm dark:text-red-300">
									There was an error loading your API keys. Please try
									refreshing the page.
								</p>
							</div>
						) : firstApiKey ? (
							<div className="rounded-lg border bg-green-50 p-4 dark:bg-green-950/20">
								<div className="flex items-center gap-2 text-green-700 dark:text-green-400">
									<Check className="h-4 w-4" />
									<span className="font-medium text-sm">
										API Key Found: {firstApiKey.key_preview}...
									</span>
								</div>
								<p className="mt-2 text-green-600 text-sm dark:text-green-300">
									Great! You already have an API key set up for this project.
								</p>
							</div>
						) : (
							<div className="rounded-lg border bg-orange-50 p-4 dark:bg-orange-950/20">
								<div className="flex items-center gap-2 text-orange-700 dark:text-orange-400">
									<ExternalLink className="h-4 w-4" />
									<span className="font-medium text-sm">No API Key Found</span>
								</div>
								<p className="mt-2 text-orange-600 text-sm dark:text-orange-300">
									You'll need to create an API key first.
								</p>
								<div className="mt-3">
									<Button size="sm" asChild>
										<Link
											href={`/api-platform/organizations/${orgId}/projects/${projectId}/api-keys`}
										>
											Create API Key
										</Link>
									</Button>
								</div>
							</div>
						)}
					</CardContent>
				</Card>

				{/* Step 2: Make Your First Request */}
				<Card>
					<CardHeader>
						<div className="flex items-center gap-3">
							<div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary font-bold text-primary-foreground text-sm">
								2
							</div>
							<div>
								<CardTitle>Make Your First Request</CardTitle>
								<CardDescription>
									The Adaptive API is fully compatible with OpenAI's API
								</CardDescription>
							</div>
						</div>
					</CardHeader>
					<CardContent>
						<Tabs defaultValue="curl" className="w-full">
							<TabsList className="grid w-full grid-cols-3">
								<TabsTrigger value="curl">cURL</TabsTrigger>
								<TabsTrigger value="javascript">JavaScript</TabsTrigger>
								<TabsTrigger value="python">Python</TabsTrigger>
							</TabsList>

							<TabsContent value="curl" className="mt-4">
								<CodeBlock
									code={curlExample}
									language="bash"
									title="cURL Request"
								/>
							</TabsContent>

							<TabsContent value="javascript" className="mt-4">
								<div className="space-y-4">
									<div className="rounded-lg border bg-blue-50 p-3 dark:bg-blue-950/20">
										<p className="font-medium text-blue-900 text-sm dark:text-blue-100">
											Install the OpenAI SDK:
										</p>
										<code className="mt-1 block text-blue-700 text-sm dark:text-blue-300">
											npm install openai
										</code>
									</div>
									<CodeBlock
										code={jsExample}
										language="javascript"
										title="JavaScript/Node.js"
									/>
								</div>
							</TabsContent>

							<TabsContent value="python" className="mt-4">
								<div className="space-y-4">
									<div className="rounded-lg border bg-blue-50 p-3 dark:bg-blue-950/20">
										<p className="font-medium text-blue-900 text-sm dark:text-blue-100">
											Install the OpenAI SDK:
										</p>
										<code className="mt-1 block text-blue-700 text-sm dark:text-blue-300">
											pip install openai
										</code>
									</div>
									<CodeBlock
										code={pythonExample}
										language="python"
										title="Python"
									/>
								</div>
							</TabsContent>
						</Tabs>
					</CardContent>
				</Card>

				{/* Step 3: Response Format */}
				<Card>
					<CardHeader>
						<div className="flex items-center gap-3">
							<div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary font-bold text-primary-foreground text-sm">
								3
							</div>
							<div>
								<CardTitle>Understanding the Response</CardTitle>
								<CardDescription>
									Here's what a typical API response looks like
								</CardDescription>
							</div>
						</div>
					</CardHeader>
					<CardContent>
						<CodeBlock
							code={exampleResponse}
							language="json"
							title="Example Response"
						/>
					</CardContent>
				</Card>

				{/* Key Features */}
				<Card>
					<CardHeader>
						<CardTitle>Key Features & Benefits</CardTitle>
						<CardDescription>
							What makes the Adaptive API special
						</CardDescription>
					</CardHeader>
					<CardContent>
						<div className="grid gap-4 md:grid-cols-2">
							<div className="space-y-2">
								<h4 className="font-medium">🔄 Intelligent Routing</h4>
								<p className="text-muted-foreground text-sm">
									Automatically routes requests to the best model and provider
									based on your prompt.
								</p>
							</div>
							<div className="space-y-2">
								<h4 className="font-medium">💰 Cost Optimization</h4>
								<p className="text-muted-foreground text-sm">
									Save 30-70% on API costs compared to using providers directly.
								</p>
							</div>
							<div className="space-y-2">
								<h4 className="font-medium">🛡️ Built-in Failover</h4>
								<p className="text-muted-foreground text-sm">
									Automatic failover between providers ensures high
									availability.
								</p>
							</div>
							<div className="space-y-2">
								<h4 className="font-medium">📊 Real-time Analytics</h4>
								<p className="text-muted-foreground text-sm">
									Monitor usage, costs, and performance in real-time.
								</p>
							</div>
						</div>
					</CardContent>
				</Card>

				{/* Next Steps */}
				<Card>
					<CardHeader>
						<CardTitle>Next Steps</CardTitle>
						<CardDescription>
							Ready to dive deeper? Here's what to explore next
						</CardDescription>
					</CardHeader>
					<CardContent>
						<div className="grid gap-4 md:grid-cols-2">
							<Link
								href={`/api-platform/organizations/${orgId}/projects/${projectId}/api-keys`}
							>
								<div className="rounded-lg border p-4 transition-colors hover:bg-accent">
									<div className="flex items-center gap-2">
										<div className="h-2 w-2 rounded-full bg-blue-500" />
										<h4 className="font-medium">Manage API Keys</h4>
									</div>
									<p className="mt-2 text-muted-foreground text-sm">
										Create, rotate, and manage your API keys
									</p>
								</div>
							</Link>
							<Link
								href={`/api-platform/organizations/${orgId}/projects/${projectId}`}
							>
								<div className="rounded-lg border p-4 transition-colors hover:bg-accent">
									<div className="flex items-center gap-2">
										<div className="h-2 w-2 rounded-full bg-green-500" />
										<h4 className="font-medium">View Dashboard</h4>
									</div>
									<p className="mt-2 text-muted-foreground text-sm">
										Monitor usage and performance metrics
									</p>
								</div>
							</Link>
							<a
								href="https://docs.adaptive.so"
								target="_blank"
								rel="noopener noreferrer"
							>
								<div className="rounded-lg border p-4 transition-colors hover:bg-accent">
									<div className="flex items-center gap-2">
										<div className="h-2 w-2 rounded-full bg-purple-500" />
										<h4 className="font-medium">Full Documentation</h4>
										<ExternalLink className="h-4 w-4 text-muted-foreground" />
									</div>
									<p className="mt-2 text-muted-foreground text-sm">
										Comprehensive API documentation and guides
									</p>
								</div>
							</a>
							<a
								href="https://github.com/adaptive/examples"
								target="_blank"
								rel="noopener noreferrer"
							>
								<div className="rounded-lg border p-4 transition-colors hover:bg-accent">
									<div className="flex items-center gap-2">
										<div className="h-2 w-2 rounded-full bg-orange-500" />
										<h4 className="font-medium">Code Examples</h4>
										<ExternalLink className="h-4 w-4 text-muted-foreground" />
									</div>
									<p className="mt-2 text-muted-foreground text-sm">
										Example projects and integrations
									</p>
								</div>
							</a>
						</div>
					</CardContent>
				</Card>

				<Separator />

				{/* Support Section */}
				<div className="text-center">
					<h3 className="mb-2 font-medium text-lg">Need Help?</h3>
					<p className="mb-4 text-muted-foreground">
						Our team is here to help you get started
					</p>
					<div className="flex flex-wrap justify-center gap-3">
						<Button variant="outline" asChild>
							<a
								href="https://docs.adaptive.so"
								target="_blank"
								rel="noopener noreferrer"
							>
								<ExternalLink className="mr-2 h-4 w-4" />
								Documentation
							</a>
						</Button>
						<Button variant="outline" asChild>
							<a
								href="mailto:support@adaptive.so"
								target="_blank"
								rel="noopener noreferrer"
							>
								Email Support
							</a>
						</Button>
						<Button variant="outline" asChild>
							<a
								href="https://discord.gg/adaptive"
								target="_blank"
								rel="noopener noreferrer"
							>
								Join Discord
							</a>
						</Button>
					</div>
				</div>
			</div>
		</div>
	);
}
