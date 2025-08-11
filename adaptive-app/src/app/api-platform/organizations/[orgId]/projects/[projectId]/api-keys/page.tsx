"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Check, Copy, Edit, Plus, Trash2 } from "lucide-react";
import { useParams } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	CodeBlock,
	CodeBlockCode,
	CodeBlockGroup,
} from "@/components/ui/code-block";
import { CopyButton } from "@/components/ui/copy-button";
import {
	Dialog,
	DialogContent,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import {
	Form,
	FormControl,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useCreateProjectApiKey } from "@/hooks/api_keys/use-create-project-api-key";
import { useDeleteProjectApiKey } from "@/hooks/api_keys/use-delete-project-api-key";
import { useProjectApiKeys } from "@/hooks/api_keys/use-project-api-keys";
import { api } from "@/trpc/react";

const formSchema = z.object({
	name: z.string().min(2, { message: "Name must be at least 2 characters." }),
	description: z.string().optional(),
});

const API_BASE_URL =
	process.env.NEXT_PUBLIC_URL ??
	(typeof window !== "undefined" ? window.location.origin : "");

export default function ApiKeysPage() {
	const params = useParams();
	const projectId = params.projectId as string;

	const [showCreateDialog, setShowCreateDialog] = useState(false);
	const [showApiKeyModal, setShowApiKeyModal] = useState(false);
	const [newApiKey, setNewApiKey] = useState<string | null>(null);
	const [copiedApiKey, setCopiedApiKey] = useState(false);

	const { data: apiKeys = [], isLoading, error } = useProjectApiKeys(projectId);
	const createApiKey = useCreateProjectApiKey();
	const deleteApiKey = useDeleteProjectApiKey();
	const revealApiKey = api.api_keys.revealApiKey.useMutation();

	const form = useForm<z.infer<typeof formSchema>>({
		resolver: zodResolver(formSchema),
		defaultValues: {
			name: "",
			description: "",
		},
	});

	const handleCreateApiKey = (values: z.infer<typeof formSchema>) => {
		createApiKey.mutate(
			{
				name: values.name,
				projectId,
				status: "active",
			},
			{
				onSuccess: (data) => {
					setShowCreateDialog(false);
					form.reset();

					// Use the reveal token to get the full API key
					revealApiKey.mutate(
						{ token: data.reveal_token },
						{
							onSuccess: (revealData) => {
								setNewApiKey(revealData.full_api_key);
								setShowApiKeyModal(true);
							},
							onError: (error) => {
								console.error("Failed to reveal API key:", error);
							},
						},
					);
				},
			},
		);
	};

	const handleCopyApiKey = async () => {
		if (newApiKey) {
			try {
				await navigator.clipboard.writeText(newApiKey);
				setCopiedApiKey(true);
				setTimeout(() => setCopiedApiKey(false), 2000);
			} catch (err) {
				console.error("Failed to copy:", err);
			}
		}
	};

	const handleCloseApiKeyModal = () => {
		setShowApiKeyModal(false);
		setNewApiKey(null);
		setCopiedApiKey(false);
	};

	const handleDeleteApiKey = (id: string) => {
		deleteApiKey.mutate({ id });
	};

	if (isLoading) {
		return (
			<div className="flex min-h-[400px] items-center justify-center">
				<div className="text-center">
					<div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
					<p className="text-muted-foreground">Loading API keys...</p>
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="flex min-h-[400px] items-center justify-center">
				<div className="text-center">
					<h3 className="mb-2 font-medium text-foreground text-lg">
						Failed to load API keys
					</h3>
					<p className="mb-4 text-muted-foreground">{error.message}</p>
				</div>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<h1 className="font-bold text-2xl text-foreground">API keys</h1>
				<Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
					<DialogTrigger asChild>
						<Button className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
							<Plus className="h-4 w-4" />
							Create new secret key
						</Button>
					</DialogTrigger>
					<DialogContent className="max-w-md">
						<DialogHeader>
							<DialogTitle>Create new secret key</DialogTitle>
						</DialogHeader>
						<div className="space-y-4">
							<Form {...form}>
								<form
									onSubmit={form.handleSubmit(handleCreateApiKey)}
									className="space-y-4"
								>
									<FormField
										control={form.control}
										name="name"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Name</FormLabel>
												<FormControl>
													<Input
														id="name"
														placeholder="My test key"
														{...field}
													/>
												</FormControl>
												<FormMessage />
											</FormItem>
										)}
									/>
									<FormField
										control={form.control}
										name="description"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Description (optional)</FormLabel>
												<FormControl>
													<Textarea
														id="description"
														placeholder="What's this key for?"
														rows={3}
														{...field}
													/>
												</FormControl>
												<FormMessage />
											</FormItem>
										)}
									/>
									<div className="flex justify-end gap-2">
										<Button
											variant="outline"
											type="button"
											onClick={() => setShowCreateDialog(false)}
										>
											Cancel
										</Button>
										<Button
											type="submit"
											disabled={!form.watch("name") || createApiKey.isPending}
											className="bg-primary hover:bg-primary/90"
										>
											{createApiKey.isPending
												? "Creating..."
												: "Create secret key"}
										</Button>
									</div>
								</form>
							</Form>
						</div>
					</DialogContent>
				</Dialog>
			</div>

			{/* Description */}
			<div className="space-y-3 text-muted-foreground text-sm">
				<p>
					As an owner of this project, you can view and manage all API keys in
					this project.
				</p>
				<p>
					Do not share your API key with others or expose it in the browser or
					other client-side code. To protect your account's security, Adaptive
					may automatically disable any API key that has leaked publicly.
				</p>
				<p>
					View usage per API key on the{" "}
					<Button
						variant="link"
						className="h-auto p-0 text-primary hover:text-primary/80"
					>
						Usage page
					</Button>
					.
				</p>
			</div>

			{/* Table */}
			<div className="overflow-hidden rounded-lg border border-border bg-card">
				<div className="overflow-x-auto">
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Name</TableHead>
								<TableHead>Secret Key</TableHead>
								<TableHead>Created</TableHead>
								<TableHead>Expires</TableHead>
								<TableHead>Created By</TableHead>
								<TableHead>Status</TableHead>
								<TableHead className="text-right">Actions</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{apiKeys.map((apiKey) => (
								<TableRow key={apiKey.id} className="hover:bg-muted/50">
									<TableCell>{apiKey.name}</TableCell>
									<TableCell className="font-mono text-muted-foreground text-sm">
										{apiKey.key_preview}...
									</TableCell>
									<TableCell className="text-muted-foreground text-sm">
										{new Date(apiKey.created_at).toLocaleDateString("en-US", {
											year: "numeric",
											month: "short",
											day: "numeric",
										})}
									</TableCell>
									<TableCell className="text-muted-foreground text-sm">
										{apiKey.expires_at
											? new Date(apiKey.expires_at).toLocaleDateString(
													"en-US",
													{
														year: "numeric",
														month: "short",
														day: "numeric",
													},
												)
											: "Never"}
									</TableCell>
									<TableCell className="text-muted-foreground text-sm">
										{apiKey.user_id}
									</TableCell>
									<TableCell className="text-muted-foreground text-sm">
										<span
											className={`inline-flex items-center rounded-full px-2 py-1 font-medium text-xs ${
												apiKey.status === "active"
													? "border border-success/20 bg-success/10 text-success"
													: "border border-destructive/20 bg-destructive/10 text-destructive"
											}`}
										>
											{apiKey.status}
										</span>
									</TableCell>
									<TableCell className="text-right font-medium text-sm">
										<div className="flex items-center justify-end gap-2">
											<Button
												variant="ghost"
												size="sm"
												disabled
												className="h-auto cursor-not-allowed p-1 text-muted-foreground/50"
												aria-disabled="true"
												aria-describedby="edit-button-disabled-description"
											>
												<Edit className="h-4 w-4" />
												<span
													id="edit-button-disabled-description"
													className="sr-only"
												>
													Editing is disabled because the API key is inactive.
												</span>
											</Button>
											<Button
												variant="ghost"
												size="sm"
												onClick={() => handleDeleteApiKey(apiKey.id)}
												disabled={deleteApiKey.isPending}
												className="h-auto p-1 text-muted-foreground hover:text-destructive"
											>
												<Trash2 className="h-4 w-4" />
											</Button>
										</div>
									</TableCell>
								</TableRow>
							))}
						</TableBody>
					</Table>
				</div>
			</div>

			{/* API Key Display Modal */}
			<Dialog open={showApiKeyModal} onOpenChange={setShowApiKeyModal}>
				<DialogContent className="!max-w-7xl sm:!max-w-7xl max-h-[90vh] w-[98vw] overflow-y-auto sm:w-[90vw] lg:w-[85vw] xl:w-[80vw]">
					<DialogHeader>
						<DialogTitle>Your API key is ready!</DialogTitle>
					</DialogHeader>
					<div className="space-y-4 px-1 sm:px-2 lg:px-0">
						{/* API Key Section */}
						<div className="space-y-3">
							<div className="space-y-2">
								<p className="font-semibold text-muted-foreground text-sm">
									Please save this API key somewhere safe and accessible. For
									security reasons, you won't be able to view it again through
									your account. If you lose this API key, you'll need to
									generate a new one.
								</p>
							</div>
							<div className="space-y-2">
								<div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
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
									<code className="break-all font-mono text-sm">
										{newApiKey}
									</code>
								</div>
							</div>
						</div>

						<Separator />

						{/* Quickstart Section */}
						<div className="space-y-4">
							<div>
								<h3 className="flex items-center gap-2 font-semibold text-lg">
									🚀 Quick Start
								</h3>
								<p className="text-muted-foreground text-sm">
									Test your new API key with these examples
								</p>
							</div>

							<Tabs defaultValue="curl" className="w-full">
								<TabsList className="grid h-auto w-full grid-cols-3">
									<TabsTrigger value="curl" className="text-xs sm:text-sm">
										cURL
									</TabsTrigger>
									<TabsTrigger
										value="javascript"
										className="text-xs sm:text-sm"
									>
										JavaScript
									</TabsTrigger>
									<TabsTrigger value="python" className="text-xs sm:text-sm">
										Python
									</TabsTrigger>
								</TabsList>

								<TabsContent value="curl" className="mt-4">
									<CodeBlock>
										<CodeBlockGroup className="border-b px-3 py-2 sm:px-4 lg:px-6">
											<span className="font-medium text-xs sm:text-sm">
												Test with cURL
											</span>
											<div className="flex items-center gap-1 sm:gap-2">
												<Badge variant="secondary" className="text-xs">
													bash
												</Badge>
												<CopyButton
													content={`curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${newApiKey}" \\
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
  }'`}
													copyMessage="cURL command copied to clipboard!"
												/>
											</div>
										</CodeBlockGroup>
										<CodeBlockCode
											code={`curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${newApiKey}" \\
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
  }'`}
											language="bash"
										/>
									</CodeBlock>
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
										<CodeBlock>
											<CodeBlockGroup className="border-b px-3 py-2 sm:px-4 lg:px-6">
												<span className="font-medium text-xs sm:text-sm">
													JavaScript/Node.js
												</span>
												<div className="flex items-center gap-1 sm:gap-2">
													<Badge variant="secondary" className="text-xs">
														javascript
													</Badge>
													<CopyButton
														content={`import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: '${newApiKey}',
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

main();`}
														copyMessage="JavaScript code copied to clipboard!"
													/>
												</div>
											</CodeBlockGroup>
											<CodeBlockCode
												code={`import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: '${newApiKey}',
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

main();`}
												language="javascript"
											/>
										</CodeBlock>
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
										<CodeBlock>
											<CodeBlockGroup className="border-b px-3 py-2 sm:px-4 lg:px-6">
												<span className="font-medium text-xs sm:text-sm">
													Python
												</span>
												<div className="flex items-center gap-1 sm:gap-2">
													<Badge variant="secondary" className="text-xs">
														python
													</Badge>
													<CopyButton
														content={`from openai import OpenAI

client = OpenAI(
    api_key="${newApiKey}",
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

print(completion.choices[0].message.content)`}
														copyMessage="Python code copied to clipboard!"
													/>
												</div>
											</CodeBlockGroup>
											<CodeBlockCode
												code={`from openai import OpenAI

client = OpenAI(
    api_key="${newApiKey}",
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

print(completion.choices[0].message.content)`}
												language="python"
											/>
										</CodeBlock>
									</div>
								</TabsContent>
							</Tabs>
						</div>

						<div className="flex justify-end pt-2">
							<Button
								onClick={handleCloseApiKeyModal}
								className="bg-primary hover:bg-primary/90"
								size="lg"
							>
								Done
							</Button>
						</div>
					</div>
				</DialogContent>
			</Dialog>
		</div>
	);
}
