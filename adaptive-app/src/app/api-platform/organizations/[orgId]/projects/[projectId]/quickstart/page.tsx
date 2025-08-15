"use client";

import { AlertCircle, ArrowLeft, Check, ExternalLink } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	CodeBlock,
	CodeBlockCode,
	CodeBlockGroup,
} from "@/components/ui/code-block";
import { CopyButton } from "@/components/ui/copy-button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/trpc/react";

const EXAMPLE_API_KEY = "your_api_key_here_placeholder_12345";
const API_BASE_URL =
	process.env.NEXT_PUBLIC_URL ?? "https://www.llmadaptive.uk";

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
	const exampleKey = EXAMPLE_API_KEY;

	const CustomCodeBlock = ({
		code,
		language,
		title,
	}: {
		code: string;
		language: string;
		title?: string;
	}) => {
		return (
			<CodeBlock>
				{title && (
					<CodeBlockGroup className="border-b px-4 py-2">
						<span className="font-medium text-sm">{title}</span>
						<div className="flex items-center gap-2">
							<Badge variant="secondary" className="text-xs">
								{language}
							</Badge>
							<CopyButton
								content={code}
								copyMessage={`${title || "Code"} copied to clipboard!`}
							/>
						</div>
					</CodeBlockGroup>
				)}
				<CodeBlockCode code={code} language={language} />
			</CodeBlock>
		);
	};

	const curlExample = `curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
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
  apiKey: '${exampleKey}',
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
								<CustomCodeBlock
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
									<CustomCodeBlock
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
									<CustomCodeBlock
										code={pythonExample}
										language="python"
										title="Python"
									/>
								</div>
							</TabsContent>
						</Tabs>
					</CardContent>
				</Card>

				{/* Step 3: Additional Endpoints */}
				<Card>
					<CardHeader>
						<div className="flex items-center gap-3">
							<div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary font-bold text-primary-foreground text-sm">
								3
							</div>
							<div>
								<CardTitle>Additional Endpoints</CardTitle>
								<CardDescription>
									Explore advanced features of the Adaptive API beyond chat
									completions
								</CardDescription>
							</div>
						</div>
					</CardHeader>
					<CardContent>
						<Tabs defaultValue="select-model" className="w-full">
							<TabsList className="grid w-full grid-cols-2">
								<TabsTrigger value="select-model">Model Selection</TabsTrigger>
								<TabsTrigger value="providers">Provider Info</TabsTrigger>
							</TabsList>

							<TabsContent value="select-model" className="mt-4">
								<div className="space-y-4">
									<div className="rounded-lg border bg-blue-50 p-4 dark:bg-blue-950/20">
										<h4 className="font-medium text-blue-900 text-sm dark:text-blue-100">
											Model Selection API
										</h4>
										<p className="mt-1 text-blue-700 text-sm dark:text-blue-300">
											Let the AI service intelligently select the best model and
											provider for your request without executing it.
										</p>
									</div>

									<CustomCodeBlock
										code={`curl -X POST "${API_BASE_URL}/api/v1/select-model" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Write a comprehensive analysis of quantum computing"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 2000
  }'`}
										language="bash"
										title="Select Best Model - cURL"
									/>

									<div className="rounded-lg border bg-gray-50 p-4 dark:bg-gray-900/20">
										<h5 className="font-medium text-gray-900 text-sm dark:text-gray-100">
											Example Response:
										</h5>
										<CustomCodeBlock
											code={`{
  "request": {
    "messages": [...],
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "metadata": {
    "provider": "openai",
    "model": "gpt-4o",
    "reasoning": "Complex analytical task requires high-quality reasoning",
    "cost_per_1m_tokens": 15.0,
    "complexity": "high",
    "cache_source": "ml_classifier"
  }
}`}
											language="json"
											title="Response"
										/>
									</div>
								</div>
							</TabsContent>

							<TabsContent value="providers" className="mt-4">
								<div className="space-y-4">
									<div className="rounded-lg border bg-green-50 p-4 dark:bg-green-950/20">
										<h4 className="font-medium text-green-900 text-sm dark:text-green-100">
											Provider Information API
										</h4>
										<p className="mt-1 text-green-700 text-sm dark:text-green-300">
											Get real-time information about available providers,
											models, and their current status.
										</p>
									</div>

									<CustomCodeBlock
										code={`curl -X GET "${API_BASE_URL}/api/v1/providers" \\
  -H "Authorization: Bearer ${exampleKey}"`}
										language="bash"
										title="List All Providers - cURL"
									/>

									<CustomCodeBlock
										code={`curl -X GET "${API_BASE_URL}/api/v1/providers/openai" \\
  -H "Authorization: Bearer ${exampleKey}"`}
										language="bash"
										title="Get Specific Provider - cURL"
									/>

									<div className="rounded-lg border bg-gray-50 p-4 dark:bg-gray-900/20">
										<h5 className="font-medium text-gray-900 text-sm dark:text-gray-100">
											Example Provider Response:
										</h5>
										<CustomCodeBlock
											code={`{
  "id": "openai",
  "name": "OpenAI",
  "status": "active",
  "models": [
    {
      "id": "gpt-4o",
      "name": "GPT-4o",
      "cost_per_1m_input_tokens": 2.5,
      "cost_per_1m_output_tokens": 10.0,
      "max_tokens": 128000,
      "supports_streaming": true
    },
    {
      "id": "gpt-4o-mini",
      "name": "GPT-4o Mini", 
      "cost_per_1m_input_tokens": 0.15,
      "cost_per_1m_output_tokens": 0.6,
      "max_tokens": 128000,
      "supports_streaming": true
    }
  ]
}`}
											language="json"
											title="Response"
										/>
									</div>
								</div>
							</TabsContent>
						</Tabs>
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
							<div className="cursor-not-allowed rounded-lg border p-4 opacity-60">
								<div className="flex items-center justify-between">
									<div className="flex items-center gap-2">
										<div className="h-2 w-2 rounded-full bg-purple-500" />
										<h4 className="font-medium">Full Documentation</h4>
									</div>
									<Badge variant="secondary" className="text-xs">
										Coming Soon
									</Badge>
								</div>
								<p className="mt-2 text-muted-foreground text-sm">
									Comprehensive API documentation and guides
								</p>
							</div>
							<div className="cursor-not-allowed rounded-lg border p-4 opacity-60">
								<div className="flex items-center justify-between">
									<div className="flex items-center gap-2">
										<div className="h-2 w-2 rounded-full bg-orange-500" />
										<h4 className="font-medium">Code Examples</h4>
									</div>
									<Badge variant="secondary" className="text-xs">
										Coming Soon
									</Badge>
								</div>
								<p className="mt-2 text-muted-foreground text-sm">
									Example projects and integrations
								</p>
							</div>
						</div>
					</CardContent>
				</Card>

				<Separator />
			</div>
		</div>
	);
}
