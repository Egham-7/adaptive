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

export default function ChatCompletionsPage() {
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

	const curlExample = `curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": false
  }'`;

	const curlStreamingExample = `curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${exampleKey}" \\
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Write a short story about AI"
      }
    ],
    "stream": true,
    "temperature": 0.8,
    "max_tokens": 500
  }'`;

	const jsExample = `// Using fetch API
const response = await fetch('${API_BASE_URL}/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ${exampleKey}'
  },
  body: JSON.stringify({
    messages: [
      {
        role: 'user',
        content: 'Explain quantum computing in simple terms'
      }
    ],
    temperature: 0.7,
    max_tokens: 1000,
    stream: false
  })
});

const result = await response.json();
console.log(result.choices[0].message.content);`;

	const jsStreamingExample = `// Streaming with fetch API
const response = await fetch('${API_BASE_URL}/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ${exampleKey}'
  },
  body: JSON.stringify({
    messages: [
      { role: 'user', content: 'Write a short story about AI' }
    ],
    stream: true,
    temperature: 0.8,
    max_tokens: 500
  })
});

if (!response.ok) {
  throw new Error(\`Request failed with \${response.status}: \${await response.text()}\`);
}
if (!response.body) {
  throw new Error('ReadableStream not supported in this environment');
}

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
let finished = false;

try {
  while (!finished) {
    const { done, value } = await reader.read();
    if (done) {
      // flush any remaining bytes and exit loop
      buffer += decoder.decode();
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\\n');
    buffer = lines.pop() ?? '';

    for (const raw of lines) {
      const line = raw.trim();
      if (!line || !line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') {
        finished = true;
        break;
      }
      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) console.log(content);
      } catch (err) {
        console.error('Failed to parse SSE chunk:', err);
      }
    }
  }
} finally {
  reader.releaseLock();
}`;

	const pythonExample = `import requests

response = requests.post(
    '${API_BASE_URL}/api/v1/chat/completions',
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${exampleKey}'
    },
    json={
        'messages': [
            {
                'role': 'user',
                'content': 'Explain quantum computing in simple terms'
            }
        ],
        'temperature': 0.7,
        'max_tokens': 1000,
        'stream': False
    }
)

result = response.json()
print(result['choices'][0]['message']['content'])`;

	const pythonStreamingExample = `import requests

response = requests.post(
    '${API_BASE_URL}/api/v1/chat/completions',
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${exampleKey}'
    },
    json={
        'messages': [
            {
                'role': 'user',
                'content': 'Write a short story about AI'
            }
        ],
        'stream': True,
        'temperature': 0.8,
        'max_tokens': 500
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        if line_text.startswith('data: '):
            data = line_text[6:]
            if data == '[DONE]':
                break
            try:
                import json
                parsed = json.loads(data)
                content = parsed['choices'][0]['delta'].get('content')
                if content:
                    print(content, end='')
            except json.JSONDecodeError:
                # Ignore malformed lines and continue streaming
                pass`;

	const exampleResponse = `{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-4o-mini",
  "provider": "openai",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is like having a super-powered computer that can solve certain problems much faster than traditional computers..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 150,
    "total_tokens": 170
  }
}`;

	const streamingResponse = `data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4o-mini","provider":"openai","choices":[{"index":0,"delta":{"role":"assistant","content":"Quantum"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4o-mini","provider":"openai","choices":[{"index":0,"delta":{"content":" computing"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4o-mini","provider":"openai","choices":[{"index":0,"delta":{"content":" is like"},"finish_reason":null}]}

data: [DONE]`;

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
						<div className="mb-2 flex items-center gap-3">
							<div>
								<h1 className="font-semibold text-3xl text-adaptive-burnt-gold">
									Chat Completions API
								</h1>
								<div className="mt-1 flex items-center gap-2">
									<Badge variant="outline" className="text-xs">
										POST
									</Badge>
									<code className="rounded bg-muted px-2 py-1 font-mono text-sm">
										/v1/chat/completions
									</code>
								</div>
							</div>
						</div>
						<p className="text-muted-foreground">
							Send messages to AI models and get intelligent responses
						</p>
					</div>
				</div>
			</div>

			{/* Overview */}
			<div className="mb-8 space-y-4">
				<p className="text-muted-foreground">
					The chat completions endpoint is your main interface for AI
					conversations. Adaptive intelligently selects the best model and
					provider for each request, optimizing for quality, cost, and
					performance automatically.
				</p>

				<div className="rounded-lg border bg-muted/30 p-4">
					<div className="space-y-2 text-sm">
						<div>
							<strong>OpenAI Compatible:</strong> Drop-in replacement for
							OpenAI's API
						</div>
						<div>
							<strong>Intelligent Routing:</strong> Automatic model selection
							based on prompt analysis
						</div>
						<div>
							<strong>Cost Optimized:</strong> 30-70% savings through smart
							provider selection
						</div>
					</div>
				</div>
			</div>

			{/* Features */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Key Features
					</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="grid gap-6 md:grid-cols-2">
						<div>
							<h4 className="mb-3 font-medium">✨ Smart Features</h4>
							<ul className="space-y-2 text-sm">
								<li>
									• <strong>Intelligent routing</strong> - Automatic model
									selection
								</li>
								<li>
									• <strong>Cost optimization</strong> - Best price-performance
									ratio
								</li>
								<li>
									• <strong>Fallback handling</strong> - Automatic failover
									between providers
								</li>
								<li>
									• <strong>Streaming support</strong> - Real-time response
									streaming
								</li>
								<li>
									• <strong>Function calling</strong> - Tool use and structured
									outputs
								</li>
							</ul>
						</div>
						<div>
							<h4 className="mb-3 font-medium">🔧 Technical Benefits</h4>
							<ul className="space-y-2 text-sm">
								<li>
									• <strong>OpenAI compatible</strong> - Drop-in replacement
								</li>
								<li>
									• <strong>Multi-provider</strong> - Access to 10+ AI providers
								</li>
								<li>
									• <strong>High throughput</strong> - 1000+ requests per second
								</li>
								<li>
									• <strong>Low latency</strong> - &lt;100ms routing overhead
								</li>
								<li>
									• <strong>Circuit breaking</strong> - Resilient error handling
								</li>
							</ul>
						</div>
					</div>
				</CardContent>
			</Card>

			{/* Code Examples */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Basic Examples
					</CardTitle>
					<CardDescription>Standard chat completion requests</CardDescription>
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
								title="Basic Chat Completion"
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

			{/* Streaming Examples */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Streaming Examples
					</CardTitle>
					<CardDescription>
						Real-time streaming responses for better user experience
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
								code={curlStreamingExample}
								language="bash"
								title="Streaming Chat Completion"
							/>
						</TabsContent>

						<TabsContent value="javascript" className="mt-4">
							<CustomCodeBlock
								code={jsStreamingExample}
								language="javascript"
								title="JavaScript Streaming Example"
							/>
						</TabsContent>

						<TabsContent value="python" className="mt-4">
							<CustomCodeBlock
								code={pythonStreamingExample}
								language="python"
								title="Python Streaming Example"
							/>
						</TabsContent>
					</Tabs>
				</CardContent>
			</Card>

			{/* Response Format */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Response Formats
					</CardTitle>
					<CardDescription>
						Understanding the response structure
					</CardDescription>
				</CardHeader>
				<CardContent>
					<Tabs defaultValue="regular" className="w-full">
						<TabsList className="grid w-full grid-cols-2">
							<TabsTrigger value="regular">Regular Response</TabsTrigger>
							<TabsTrigger value="streaming">Streaming Response</TabsTrigger>
						</TabsList>

						<TabsContent value="regular" className="mt-4">
							<CustomCodeBlock
								code={exampleResponse}
								language="json"
								title="Standard Response"
							/>
							<div className="mt-4 space-y-4">
								<h4 className="font-medium">Key Response Fields</h4>
								<div className="grid gap-3 text-sm">
									<div className="flex gap-3">
										<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
											choices[0].message.content
										</code>
										<span>The AI's response message</span>
									</div>
									<div className="flex gap-3">
										<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
											provider
										</code>
										<span>
											Which AI provider was selected (openai, anthropic, etc.)
										</span>
									</div>
									<div className="flex gap-3">
										<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
											model
										</code>
										<span>Specific model used for the response</span>
									</div>
									<div className="flex gap-3">
										<code className="rounded bg-muted px-2 py-1 font-mono text-xs">
											usage
										</code>
										<span>Token usage and billing information</span>
									</div>
								</div>
							</div>
						</TabsContent>

						<TabsContent value="streaming" className="mt-4">
							<CustomCodeBlock
								code={streamingResponse}
								language="bash"
								title="Server-Sent Events Stream"
							/>
							<div className="mt-4 space-y-4">
								<h4 className="font-medium">Streaming Format</h4>
								<div className="rounded-lg border border-adaptive-gold/20 bg-adaptive-gold-dust/5 p-4">
									<ul className="space-y-2 text-sm">
										<li>
											• Each line starts with <code>data: </code>
										</li>
										<li>
											• <code>delta.content</code> contains the text chunk
										</li>
										<li>
											• Stream ends with <code>data: [DONE]</code>
										</li>
										<li>
											• Parse each JSON chunk to build the complete response
										</li>
									</ul>
								</div>
							</div>
						</TabsContent>
					</Tabs>
				</CardContent>
			</Card>

			{/* Integration Pattern */}
			<Card className="mb-8 border-adaptive-gold/20">
				<CardHeader>
					<CardTitle className="text-2xl text-adaptive-burnt-gold">
						Drop-in OpenAI Replacement
					</CardTitle>
					<CardDescription>
						Use with existing OpenAI SDKs and libraries
					</CardDescription>
				</CardHeader>
				<CardContent>
					<CustomCodeBlock
						code={`// OpenAI SDK - just change the base URL
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: '${exampleKey}',
  baseURL: '${API_BASE_URL}/api/v1'  // Point to Adaptive
});

const completion = await client.chat.completions.create({
  messages: [
    { role: 'user', content: 'Hello! How can AI help businesses today?' }
  ],
  // No model parameter needed - Adaptive chooses the best one
  temperature: 0.7,
  max_tokens: 1000
});

console.log(completion.choices[0].message.content);`}
						language="javascript"
						title="Using OpenAI SDK with Adaptive"
					/>

					<div className="mt-4 rounded-lg border border-adaptive-gold/20 bg-adaptive-gold-dust/5 p-4">
						<h5 className="mb-2 font-medium text-adaptive-burnt-gold">
							🔄 Migration Benefits
						</h5>
						<ul className="space-y-1 text-sm">
							<li>
								• <strong>Zero code changes</strong> - just update the base URL
							</li>
							<li>
								• <strong>Automatic optimization</strong> - no model parameter
								needed
							</li>
							<li>
								• <strong>Cost savings</strong> - intelligent provider selection
							</li>
							<li>
								• <strong>Better reliability</strong> - built-in fallbacks
							</li>
						</ul>
					</div>
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
						href={`/api-platform/organizations/${orgId}/projects/${projectId}/examples/select-model`}
						className="hover:text-foreground"
					>
						Model Selection →
					</Link>
				</div>
			</div>
		</div>
	);
}
