"use client";

import { ArrowLeft } from "lucide-react";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/trpc/react";

const API_BASE_URL =
	process.env.NEXT_PUBLIC_URL ?? "https://www.llmadaptive.uk";

export default function ProvidersPage() {
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

	return (
		<div className="mx-auto max-w-7xl px-2 py-2 sm:px-4">
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
						<h1 className="font-semibold text-3xl text-adaptive-burnt-gold">
							Provider Management API
						</h1>
						<p className="text-muted-foreground">
							Discover, configure, and manage AI providers and their models
						</p>
					</div>
					<div className="flex items-center gap-2">
						<Badge variant="secondary">Multi-Provider</Badge>
						<Badge variant="outline">REST API</Badge>
					</div>
				</div>
			</div>

			{/* Overview */}
			<div className="mb-8 space-y-4">
				<p className="text-muted-foreground">
					The Provider API gives you complete control over AI providers in your
					project. List available providers, get detailed information, and
					manage configurations all through a unified interface.
				</p>
			</div>

			{/* Endpoints Overview */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Available Endpoints
					</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="grid gap-4 md:grid-cols-3">
						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<div className="mb-2 flex items-center gap-2">
								<Badge variant="outline" className="text-xs">
									GET
								</Badge>
								<code className="font-mono text-sm">/v1/providers</code>
							</div>
							<p className="text-muted-foreground text-sm">
								List all available providers with models and pricing
							</p>
						</div>

						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<div className="mb-2 flex items-center gap-2">
								<Badge variant="outline" className="text-xs">
									GET
								</Badge>
								<code className="font-mono text-sm">
									/v1/providers/&#123;id&#125;
								</code>
							</div>
							<p className="text-muted-foreground text-sm">
								Get detailed information about a specific provider
							</p>
						</div>

						<div className="rounded-lg border border-adaptive-gold/10 bg-adaptive-gold-dust/5 p-4">
							<div className="mb-2 flex items-center gap-2">
								<Badge variant="outline" className="text-xs">
									POST/PUT
								</Badge>
								<code className="font-mono text-sm">
									/v1/providers/&#123;id&#125;/config
								</code>
							</div>
							<p className="text-muted-foreground text-sm">
								Configure API keys and settings for providers
							</p>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Code Examples */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Quick Examples
					</CardTitle>
					<CardDescription>
						Common operations with the Provider API
					</CardDescription>
				</CardHeader>
				<CardContent>
					<Tabs defaultValue="list" className="w-full">
						<TabsList className="grid w-full grid-cols-3">
							<TabsTrigger value="list">List Providers</TabsTrigger>
							<TabsTrigger value="details">Provider Details</TabsTrigger>
							<TabsTrigger value="config">Configuration</TabsTrigger>
						</TabsList>

						<TabsContent value="list" className="mt-4">
							<CustomCodeBlock
								code={`# List all providers
curl "${API_BASE_URL}/api/v1/providers?project_id=${projectId}" \\
  -H "Authorization: Bearer ${exampleKey}"

# Response includes providers with models and config status
{
  "providers": [
    {
      "id": "prov_openai_123",
      "name": "openai", 
      "displayName": "OpenAI",
      "models": [
        {
          "name": "gpt-4o-mini",
          "inputTokenCost": 0.15,
          "outputTokenCost": 0.6
        }
      ],
      "config": { "isConfigured": true }
    }
  ],
  "count": 1
}`}
								language="bash"
								title="List All Providers"
							/>
						</TabsContent>

						<TabsContent value="details" className="mt-4">
							<CustomCodeBlock
								code={`# Get specific provider details
curl "${API_BASE_URL}/api/v1/providers/prov_openai_123" \\
  -H "Authorization: Bearer ${exampleKey}"

# Detailed response with full model capabilities
{
  "id": "prov_openai_123",
  "name": "openai",
  "displayName": "OpenAI",
  "description": "OpenAI's GPT models for general-purpose AI tasks",
  "baseUrl": "https://api.openai.com/v1",
  "models": [
    {
      "id": "model_gpt4o_123",
      "name": "gpt-4o",
      "displayName": "GPT-4o",
      "inputTokenCost": 2.5,
      "outputTokenCost": 10.0,
      "capabilities": {
        "maxContextTokens": 128000,
        "supportsFunctionCalling": true,
        "supportsVision": true,
        "taskTypes": ["general", "analysis", "coding"]
      }
    }
  ]
}`}
								language="bash"
								title="Get Provider Details"
							/>
						</TabsContent>

						<TabsContent value="config" className="mt-4">
							<CustomCodeBlock
								code={`# Configure a provider (set API key)
curl -X POST "${API_BASE_URL}/api/v1/providers/prov_openai_123/config" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
    "apiKey": "sk-your-openai-api-key-here",
    "isActive": true
  }'

# Update existing configuration
curl -X PUT "${API_BASE_URL}/api/v1/providers/prov_openai_123/config" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
    "isActive": false
  }'`}
								language="bash"
								title="Provider Configuration"
							/>
						</TabsContent>
					</Tabs>
				</CardContent>
			</Card>

			{/* Use Cases */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Common Use Cases
					</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="grid gap-6 md:grid-cols-2">
						<div>
							<h4 className="mb-3 font-medium text-adaptive-pine">
								🔍 Discovery & Analysis
							</h4>
							<ul className="space-y-2 text-sm">
								<li>• Compare pricing across all providers</li>
								<li>
									• Find models with specific capabilities (vision, function
									calling)
								</li>
								<li>• Check configuration status before using</li>
								<li>• Build cost estimation dashboards</li>
							</ul>
						</div>

						<div>
							<h4 className="mb-3 font-medium text-adaptive-slate">
								⚙️ Configuration & Management
							</h4>
							<ul className="space-y-2 text-sm">
								<li>• Set up API keys for new providers</li>
								<li>• Enable/disable providers as needed</li>
								<li>• Validate configurations work correctly</li>
								<li>• Rotate API keys securely</li>
							</ul>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* JavaScript Integration Example */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						JavaScript Integration
					</CardTitle>
					<CardDescription>
						Complete example for building a provider dashboard
					</CardDescription>
				</CardHeader>
				<CardContent>
					<CustomCodeBlock
						code={`class ProviderManager {
  constructor(apiKey, projectId) {
    this.apiKey = apiKey;
    this.projectId = projectId;
    this.baseUrl = '${API_BASE_URL}/api/v1';
  }

  // Get all providers with configuration status
  async listProviders() {
    const response = await fetch(\`\${this.baseUrl}/providers?project_id=\${this.projectId}\`, {
      headers: { 'Authorization': \`Bearer \${this.apiKey}\` }
    });
    return response.json();
  }

  // Get detailed provider information
  async getProvider(providerId) {
    const response = await fetch(\`\${this.baseUrl}/providers/\${providerId}\`, {
      headers: { 'Authorization': \`Bearer \${this.apiKey}\` }
    });
    return response.json();
  }

  // Configure a provider
  async configureProvider(providerId, config) {
    const response = await fetch(\`\${this.baseUrl}/providers/\${providerId}/config\`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': \`Bearer \${this.apiKey}\`
      },
      body: JSON.stringify(config)
    });
    return response.json();
  }

  // Build a cost comparison dashboard
  async buildCostDashboard() {
    const { providers } = await this.listProviders();
    
    const models = providers.flatMap(p => 
      p.models.map(m => ({
        provider: p.displayName,
        model: m.displayName,
        inputCost: m.inputTokenCost,
        outputCost: m.outputTokenCost,
        configured: p.config?.isConfigured || false
      }))
    );

    // Sort by cost for budget-conscious users
    return models.sort((a, b) => a.inputCost - b.inputCost);
  }
}

// Usage example
const manager = new ProviderManager('${exampleKey}', '${projectId}');
const costDashboard = await manager.buildCostDashboard();
console.log('Most affordable models:', costDashboard.slice(0, 5));`}
						language="javascript"
						title="Provider Management Class"
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
