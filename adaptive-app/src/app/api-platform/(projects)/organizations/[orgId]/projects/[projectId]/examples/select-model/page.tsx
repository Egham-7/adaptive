"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";

import { ProjectBreadcrumb } from "@/components/project-breadcrumb";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/trpc/react";

const _EXAMPLE_API_KEY = "your_api_key_here_placeholder_12345";
const API_BASE_URL =
	process.env.NEXT_PUBLIC_URL ?? "https://www.llmadaptive.uk";

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

export default function SelectModelPage() {
	const { orgId, projectId } = useParams<{
		orgId: string;
		projectId: string;
	}>();

	// Fetch project API keys for examples
	const { data: apiKeys } = api.api_keys.getByProject.useQuery({ projectId });

	const firstApiKey = apiKeys?.[0];
	const exampleKey = firstApiKey?.key_preview
		? `your_api_key_here_preview_${firstApiKey.key_preview}`
		: "your_api_key_here_preview_sk-abcd";

	const curlExample = `curl -X POST "${API_BASE_URL}/api/v1/select-model" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Write a comprehensive analysis of renewable energy trends"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 2000
  }'`;

	const jsExample = `// Using fetch API
const response = await fetch('${API_BASE_URL}/api/v1/select-model', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ${exampleKey}'
  },
  body: JSON.stringify({
    messages: [
      {
        role: 'user',
        content: 'Write a comprehensive analysis of renewable energy trends'
      }
    ],
    temperature: 0.7,
    max_tokens: 2000
  })
});

const result = await response.json();
console.log('Selected model:', result.metadata.model);
console.log('Provider:', result.metadata.provider);
console.log('Reasoning:', result.metadata.reasoning);
console.log('Cost per 1M tokens:', result.metadata.cost_per_1m_tokens);`;

	const pythonExample = `import requests

response = requests.post(
    '${API_BASE_URL}/api/v1/select-model',
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${exampleKey}'
    },
    json={
        'messages': [
            {
                'role': 'user',
                'content': 'Write a comprehensive analysis of renewable energy trends'
            }
        ],
        'temperature': 0.7,
        'max_tokens': 2000
    }
)

result = response.json()
print(f"Selected model: {result['metadata']['model']}")
print(f"Provider: {result['metadata']['provider']}")
print(f"Reasoning: {result['metadata']['reasoning']}")
print(f"Cost per 1M tokens: {result['metadata']['cost_per_1m_tokens']}")`;

	const exampleResponse = `{
  "request": {
    "messages": [
      {
        "role": "user",
        "content": "Write a comprehensive analysis of renewable energy trends"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "metadata": {
    "provider": "openai",
    "model": "gpt-4o",
    "reasoning": "Complex analytical task requiring high-quality reasoning and comprehensive analysis. GPT-4o selected for superior analytical capabilities.",
    "cost_per_1m_tokens": 15.0,
    "complexity": "high",
    "task_type": "analysis",
    "cache_source": "ml_classifier",
    "estimated_input_tokens": 1250,
    "estimated_output_tokens": 2000,
    "estimated_total_cost_usd": 0.04875
  }
}`;

	return (
		<div className="mx-auto max-w-7xl px-2 py-2 sm:px-4">
			<div className="mb-6">
				<ProjectBreadcrumb />
			</div>
			{/* Back Navigation */}
			<div className="mb-6">
				<Button variant="ghost" size="sm" asChild>
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/examples`}
					>
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Examples
					</Link>
				</Button>
			</div>

			{/* Header Section */}
			<div className="mb-8">
				<div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
					<div>
						<div className="mb-2 flex items-center gap-3">
							<div>
								<h1 className="font-semibold text-3xl text-adaptive-burnt-gold">
									Model Selection API
								</h1>
								<div className="mt-1 flex items-center gap-2">
									<Badge variant="outline" className="text-xs">
										POST
									</Badge>
									<code className="rounded bg-muted px-2 py-1 font-mono text-sm">
										/v1/select-model
									</code>
								</div>
							</div>
						</div>
						<p className="text-muted-foreground">
							Get the optimal model and provider for your request without
							executing it
						</p>
					</div>
				</div>
			</div>

			{/* Overview */}
			<div className="mb-8 space-y-4">
				<p className="text-muted-foreground">
					Returns the optimal model and provider for a given request without
					executing it. Use this endpoint to understand model selection logic,
					estimate costs, or pre-select models for batch processing.
				</p>

				<div className="rounded-lg border bg-muted/30 p-4">
					<div className="space-y-2 text-sm">
						<div>
							<strong>Input:</strong> Messages array and optional parameters
						</div>
						<div>
							<strong>Output:</strong> Model recommendation with reasoning and
							cost estimates
						</div>
						<div>
							<strong>Execution:</strong> No completion is generated
						</div>
					</div>
				</div>
			</div>

			{/* When to Use */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						When to use this endpoint
					</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="grid gap-6 md:grid-cols-2">
						<div>
							<h4 className="mb-3 font-medium">Use cases</h4>
							<ul className="space-y-2 text-sm">
								<li>
									• <strong>Cost planning</strong> - See costs before executing
								</li>
								<li>
									• <strong>Model comparison</strong> - Understand why specific
									models are chosen
								</li>
								<li>
									• <strong>Batch processing</strong> - Pre-select models for
									large jobs
								</li>
								<li>
									• <strong>Debugging</strong> - Understand model selection
									logic
								</li>
								<li>
									• <strong>Analytics</strong> - Track which models work for
									different tasks
								</li>
							</ul>
						</div>
						<div>
							<h4 className="mb-3 font-medium">Not needed for</h4>
							<ul className="space-y-2 text-muted-foreground text-sm">
								<li>
									• Regular chat - Use <code>/v1/chat/completions</code>{" "}
									directly
								</li>
								<li>• Simple requests - Automatic selection is faster</li>
								<li>• Real-time chat - Adds unnecessary latency</li>
								<li>• One-off requests - Automatic selection works great</li>
							</ul>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Code Examples */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Code Examples
					</CardTitle>
					<CardDescription>
						Try these examples to see how model selection works
					</CardDescription>
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
								title="Model Selection Request"
							/>
						</TabsContent>

						<TabsContent value="javascript" className="mt-4">
							<CustomCodeBlock
								code={jsExample}
								language="javascript"
								title="JavaScript/Node.js Example"
							/>
						</TabsContent>

						<TabsContent value="python" className="mt-4">
							<CustomCodeBlock
								code={pythonExample}
								language="python"
								title="Python Example"
							/>
						</TabsContent>
					</Tabs>
				</CardContent>
			</Card>

			{/* Response Format */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Response Format
					</CardTitle>
					<CardDescription>
						Here's what you'll get back from the model selection endpoint
					</CardDescription>
				</CardHeader>
				<CardContent>
					<CustomCodeBlock
						code={exampleResponse}
						language="json"
						title="Example Response"
					/>
					<div className="mt-4 space-y-3">
						<h4 className="font-medium">Response Fields Explained:</h4>
						<div className="grid gap-3 text-sm">
							<div className="flex gap-3">
								<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
									request
								</code>
								<span>Your original request parameters</span>
							</div>
							<div className="flex gap-3">
								<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
									provider
								</code>
								<span>
									Selected AI provider (openai, anthropic, groq, etc.)
								</span>
							</div>
							<div className="flex gap-3">
								<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
									model
								</code>
								<span>Specific model chosen for your task</span>
							</div>
							<div className="flex gap-3">
								<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
									reasoning
								</code>
								<span>Explanation of why this model was selected</span>
							</div>
							<div className="flex gap-3">
								<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
									cost_per_1m_tokens
								</code>
								<span>Pricing for this model (USD per 1M tokens)</span>
							</div>
							<div className="flex gap-3">
								<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
									estimated_total_cost_usd
								</code>
								<span>Estimated cost for your specific request</span>
							</div>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Real-world Examples */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Real-world Selection Examples
					</CardTitle>
					<CardDescription>
						See how different prompts result in different model choices
					</CardDescription>
				</CardHeader>
				<CardContent>
					<div className="grid gap-4 md:grid-cols-2">
						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<h4 className="mb-2 font-medium text-adaptive-pine">
								Simple Question → Cheaper Model
							</h4>
							<div className="mb-3 rounded bg-muted p-3">
								<code className="text-sm">"What's the capital of France?"</code>
							</div>
							<div className="space-y-1 text-sm">
								<div>
									Selected: <code>gpt-4o-mini</code>
								</div>
								<div>
									Cost: <code>$0.15/1M tokens</code>
								</div>
								<div>Reasoning: Simple factual query</div>
							</div>
						</div>

						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<h4 className="mb-2 font-medium text-adaptive-slate">
								Complex Analysis → Premium Model
							</h4>
							<div className="mb-3 rounded bg-muted p-3">
								<code className="text-sm">
									"Analyze the economic implications of renewable energy
									adoption..."
								</code>
							</div>
							<div className="space-y-1 text-sm">
								<div>
									Selected: <code>gpt-4o</code>
								</div>
								<div>
									Cost: <code>$15.0/1M tokens</code>
								</div>
								<div>
									Reasoning: Complex analytical task requiring high-quality
									reasoning
								</div>
							</div>
						</div>

						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<h4 className="mb-2 font-medium text-adaptive-grape">
								Code Generation → Specialized Model
							</h4>
							<div className="mb-3 rounded bg-muted p-3">
								<code className="text-sm">
									"Write a React component for a data table..."
								</code>
							</div>
							<div className="space-y-1 text-sm">
								<div>
									Selected: <code>deepseek-coder</code>
								</div>
								<div>
									Cost: <code>$0.14/1M tokens</code>
								</div>
								<div>
									Reasoning: Code generation task, specialized model optimal
								</div>
							</div>
						</div>

						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<h4 className="mb-2 font-medium text-adaptive-burnt-gold">
								Creative Writing → Balanced Model
							</h4>
							<div className="mb-3 rounded bg-muted p-3">
								<code className="text-sm">
									"Write a short story about time travel..."
								</code>
							</div>
							<div className="space-y-1 text-sm">
								<div>
									Selected: <code>claude-3.5-sonnet</code>
								</div>
								<div>
									Cost: <code>$3.0/1M tokens</code>
								</div>
								<div>
									Reasoning: Creative task, good balance of quality and cost
								</div>
							</div>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Integration Pattern */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Integration Pattern
					</CardTitle>
					<CardDescription>
						How to use model selection with chat completions
					</CardDescription>
				</CardHeader>
				<CardContent>
					<CustomCodeBlock
						code={`async function smartCompletion(userMessage) {
  // Step 1: Get model recommendation
  const modelSelection = await fetch('${API_BASE_URL}/api/v1/select-model', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ${exampleKey}'
    },
    body: JSON.stringify({
      messages: [{ role: 'user', content: userMessage }]
    })
  });

  const { metadata, request } = await modelSelection.json();
  console.log(\`💡 Recommended: \${metadata.provider}/\${metadata.model}\`);
  console.log(\`⚙️ Optimized params:\`, request);

  // Step 2: Execute with the recommended model (optional)
  const completion = await fetch('${API_BASE_URL}/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ${exampleKey}'
    },
    body: JSON.stringify({
      messages: [{ role: 'user', content: userMessage }]
      // Note: No model specified - will use automatic selection
    })
  });

  return completion.json();
}`}
						language="javascript"
						title="Smart Completion with Model Selection"
					/>
				</CardContent>
			</Card>

			{/* Footer */}
			<div className="mt-8 text-center">
				<Separator className="mb-6" />
				<div className="flex items-center justify-center gap-4 text-muted-foreground text-sm">
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/examples`}
						className="flex items-center gap-1 hover:text-foreground"
					>
						<ArrowLeft className="h-3 w-3" />
						All Examples
					</Link>
					<Link
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/examples/chat-completions`}
						className="hover:text-foreground"
					>
						Chat Completions →
					</Link>
				</div>
			</div>
		</div>
	);
}
