"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import {
	ArrowLeft,
	Building2,
	Check,
	CheckCircle,
	ChevronRight,
	Copy,
	FolderPlus,
	Key,
	Rocket,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import * as z from "zod";
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
import {
	Form,
	FormControl,
	FormDescription,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useCreateProjectApiKey } from "@/hooks/api_keys/use-create-project-api-key";
import { useCreateOrganization } from "@/hooks/organizations/use-create-organization";
import { useCreateProject } from "@/hooks/projects/use-create-project";
import { setLastProject } from "@/hooks/use-smart-redirect";
import { api } from "@/trpc/react";
import type {
	OrganizationCreateResponse,
	ProjectCreateResponse,
} from "@/types";

const organizationSchema = z.object({
	name: z.string().min(1, "Organization name is required"),
	description: z.string().optional(),
});

const projectSchema = z.object({
	name: z.string().min(1, "Project name is required"),
	description: z.string().optional(),
});

type OnboardingStep =
	| "welcome"
	| "organization"
	| "project"
	| "api-key"
	| "quickstart"
	| "complete";

const API_BASE_URL =
	process.env.NEXT_PUBLIC_URL ??
	(typeof window !== "undefined" ? window.location.origin : "");

export default function OnboardingPage() {
	const [currentStep, setCurrentStep] = useState<OnboardingStep>("welcome");
	const [createdOrganization, setCreatedOrganization] =
		useState<OrganizationCreateResponse | null>(null);

	const [createdProject, setCreatedProject] =
		useState<ProjectCreateResponse | null>(null);
	const [createdApiKey, setCreatedApiKey] = useState<string | null>(null);
	const [copiedApiKey, setCopiedApiKey] = useState(false);
	const router = useRouter();

	const createOrganization = useCreateOrganization();
	const createProject = useCreateProject();
	const createApiKey = useCreateProjectApiKey();
	const revealApiKey = api.api_keys.revealApiKey.useMutation();

	const organizationForm = useForm<z.infer<typeof organizationSchema>>({
		resolver: zodResolver(organizationSchema),
		defaultValues: {
			name: "",
			description: "",
		},
	});

	const projectForm = useForm<z.infer<typeof projectSchema>>({
		resolver: zodResolver(projectSchema),
		defaultValues: {
			name: "",
			description: "",
		},
	});

	const onOrganizationSubmit = (values: z.infer<typeof organizationSchema>) => {
		createOrganization.mutate(values, {
			onSuccess: (data) => {
				setCreatedOrganization(data);
				setCurrentStep("project");
			},
			onError: (error) => {
				console.error("Failed to create organization: ", error);
				toast.error(`Failed to create organization: ${error.message}`);
			},
		});
	};

	const onProjectSubmit = (values: z.infer<typeof projectSchema>) => {
		if (!createdOrganization) return;

		createProject.mutate(
			{
				...values,
				organizationId: createdOrganization.id,
			},
			{
				onSuccess: (data) => {
					setCreatedProject(data);
					// Update localStorage with the created organization and project
					if (createdOrganization) {
						setLastProject(createdOrganization.id, data.id);
					}
					setCurrentStep("api-key");
				},

				onError: (error) => {
					console.error("Failed to create project: ", error);
					toast.error(`Failed to create project: ${error.message}`);
				},
			},
		);
	};

	const handleCreateApiKey = () => {
		if (!createdProject) return;

		createApiKey.mutate(
			{
				name: "Default API Key",
				projectId: createdProject.id,
				status: "active",
			},
			{
				onSuccess: (data) => {
					// Use the reveal token to get the full API key
					revealApiKey.mutate(
						{ token: data.reveal_token },
						{
							onSuccess: (revealData) => {
								setCreatedApiKey(revealData.full_api_key);
								setCurrentStep("quickstart");
							},
							onError: (error) => {
								console.error("Failed to reveal API key:", error);
								toast.error("Failed to reveal API key");
							},
						},
					);
				},
				onError: (error) => {
					console.error("Failed to create API key:", error);
					toast.error(`Failed to create API key: ${error.message}`);
				},
			},
		);
	};

	const handleCopyApiKey = async () => {
		if (createdApiKey) {
			try {
				await navigator.clipboard.writeText(createdApiKey);
				setCopiedApiKey(true);
				setTimeout(() => setCopiedApiKey(false), 2000);
			} catch (err) {
				console.error("Failed to copy:", err);
			}
		}
	};

	const getStepNumber = (step: OnboardingStep): number => {
		switch (step) {
			case "welcome":
				return 1;
			case "organization":
				return 2;
			case "project":
				return 3;
			case "api-key":
				return 4;
			case "quickstart":
				return 5;
			case "complete":
				return 6;
			default:
				return 1;
		}
	};

	const getProgress = (): number => {
		return ((getStepNumber(currentStep) - 1) / 5) * 100;
	};

	const handleComplete = () => {
		if (createdOrganization && createdProject) {
			router.push(
				`/api-platform/organizations/${createdOrganization.id}/projects/${createdProject.id}`,
			);
		} else {
			router.push("/api-platform/organizations");
		}
	};

	const handleBack = () => {
		switch (currentStep) {
			case "organization":
				setCurrentStep("welcome");
				break;
			case "project":
				setCurrentStep("organization");
				break;
			case "api-key":
				setCurrentStep("project");
				break;
			case "quickstart":
				setCurrentStep("api-key");
				break;
			default:
				break;
		}
	};

	return (
		<div className="min-h-screen bg-gradient-to-br from-background to-muted/50">
			<div className="container mx-auto max-w-2xl px-4 py-12">
				{/* Header */}
				<div className="mb-8 text-center">
					<h1 className="mb-2 font-bold text-3xl text-foreground">
						Welcome to Adaptive AI Platform
					</h1>
					<p className="text-muted-foreground">
						Let's get you set up with your first organization and project
					</p>
				</div>

				{/* Progress */}
				<div className="mb-8">
					<Progress value={getProgress()} className="h-2" />
					<div className="mt-2 flex justify-between text-muted-foreground text-sm">
						<span>Step {getStepNumber(currentStep)} of 6</span>
						<span>{Math.round(getProgress())}% complete</span>
					</div>
				</div>

				{/* Welcome Step */}
				{currentStep === "welcome" && (
					<Card className="border-2">
						<CardHeader className="text-center">
							<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-primary/10">
								<Building2 className="h-8 w-8 text-primary" />
							</div>
							<CardTitle className="text-2xl">Let's start building</CardTitle>
							<CardDescription className="text-base">
								Organizations help you manage your AI projects and collaborate
								with your team. You can always create more organizations later.
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
												Individual AI projects with their own API keys and
												analytics
											</p>
										</div>
									</div>
								</div>
								<Button
									onClick={() => setCurrentStep("organization")}
									className="w-full"
									size="lg"
								>
									Get Started
									<ChevronRight className="ml-2 h-4 w-4" />
								</Button>
							</div>
						</CardContent>
					</Card>
				)}

				{/* Organization Step */}
				{currentStep === "organization" && (
					<Card className="border-2">
						<CardHeader>
							<div className="mb-4 flex items-center gap-2">
								<Button
									variant="ghost"
									size="sm"
									onClick={handleBack}
									className="p-2"
								>
									<ArrowLeft className="h-4 w-4" />
								</Button>
								<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
									<Building2 className="h-5 w-5 text-primary" />
								</div>
							</div>
							<CardTitle className="text-xl">
								Create your organization
							</CardTitle>
							<CardDescription>
								This will be your main workspace for managing AI projects and
								team collaboration.
							</CardDescription>
						</CardHeader>
						<CardContent>
							<Form {...organizationForm}>
								<form
									onSubmit={organizationForm.handleSubmit(onOrganizationSubmit)}
									className="space-y-6"
								>
									<FormField
										control={organizationForm.control}
										name="name"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Organization Name</FormLabel>
												<FormControl>
													<Input
														placeholder="e.g., Acme AI Solutions"
														{...field}
													/>
												</FormControl>
												<FormDescription>
													Choose a name that represents your company or team
												</FormDescription>
												<FormMessage />
											</FormItem>
										)}
									/>
									<FormField
										control={organizationForm.control}
										name="description"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Description (optional)</FormLabel>
												<FormControl>
													<Textarea
														placeholder="What does your organization do?"
														rows={3}
														{...field}
													/>
												</FormControl>
												<FormDescription>
													Help your team understand the purpose of this
													organization
												</FormDescription>
												<FormMessage />
											</FormItem>
										)}
									/>
									<div className="flex justify-end gap-3">
										<Button
											type="button"
											variant="outline"
											onClick={handleBack}
										>
											Back
										</Button>
										<Button
											type="submit"
											disabled={createOrganization.isPending}
											className="min-w-32"
										>
											{createOrganization.isPending
												? "Creating..."
												: "Continue"}
											<ChevronRight className="ml-2 h-4 w-4" />
										</Button>
									</div>
								</form>
							</Form>
						</CardContent>
					</Card>
				)}

				{/* Project Step */}
				{currentStep === "project" && (
					<Card className="border-2">
						<CardHeader>
							<div className="mb-4 flex items-center gap-2">
								<Button
									variant="ghost"
									size="sm"
									onClick={handleBack}
									className="p-2"
								>
									<ArrowLeft className="h-4 w-4" />
								</Button>
								<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
									<FolderPlus className="h-5 w-5 text-primary" />
								</div>
							</div>
							<CardTitle className="text-xl">
								Create your first project
							</CardTitle>
							<CardDescription>
								Projects help you organize your AI work with dedicated API keys,
								usage tracking, and analytics.
							</CardDescription>
						</CardHeader>
						<CardContent>
							<Form {...projectForm}>
								<form
									onSubmit={projectForm.handleSubmit(onProjectSubmit)}
									className="space-y-6"
								>
									<FormField
										control={projectForm.control}
										name="name"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Project Name</FormLabel>
												<FormControl>
													<Input
														placeholder="e.g., Customer Support AI"
														{...field}
													/>
												</FormControl>
												<FormDescription>
													Choose a descriptive name for your AI project
												</FormDescription>
												<FormMessage />
											</FormItem>
										)}
									/>
									<FormField
										control={projectForm.control}
										name="description"
										render={({ field }) => (
											<FormItem>
												<FormLabel>Description (optional)</FormLabel>
												<FormControl>
													<Textarea
														placeholder="What will this project be used for?"
														rows={3}
														{...field}
													/>
												</FormControl>
												<FormDescription>
													Describe the purpose and goals of this project
												</FormDescription>
												<FormMessage />
											</FormItem>
										)}
									/>
									<div className="flex justify-end gap-3">
										<Button
											type="button"
											variant="outline"
											onClick={handleBack}
										>
											Back
										</Button>
										<Button
											type="submit"
											disabled={createProject.isPending}
											className="min-w-32"
										>
											{createProject.isPending ? "Creating..." : "Continue"}
											<ChevronRight className="ml-2 h-4 w-4" />
										</Button>
									</div>
								</form>
							</Form>
						</CardContent>
					</Card>
				)}

				{/* API Key Step */}
				{currentStep === "api-key" && (
					<Card className="border-2">
						<CardHeader>
							<div className="mb-4 flex items-center gap-2">
								<Button
									variant="ghost"
									size="sm"
									onClick={handleBack}
									className="p-2"
								>
									<ArrowLeft className="h-4 w-4" />
								</Button>
								<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
									<Key className="h-5 w-5 text-primary" />
								</div>
							</div>
							<CardTitle className="text-xl">Generate your API key</CardTitle>
							<CardDescription>
								Create your first API key to start making requests to the
								Adaptive AI Platform.
							</CardDescription>
						</CardHeader>
						<CardContent>
							<div className="space-y-6">
								<div className="rounded-lg bg-muted/50 p-4">
									<h4 className="mb-2 font-medium">What you'll get:</h4>
									<ul className="space-y-1 text-muted-foreground text-sm">
										<li>• A secure API key for your project</li>
										<li>• OpenAI-compatible endpoint access</li>
										<li>• Usage tracking and analytics</li>
										<li>• Cost monitoring and optimization</li>
									</ul>
								</div>
								<div className="flex justify-end gap-3">
									<Button type="button" variant="outline" onClick={handleBack}>
										Back
									</Button>
									<Button
										onClick={handleCreateApiKey}
										disabled={createApiKey.isPending}
										className="min-w-32"
									>
										{createApiKey.isPending
											? "Creating..."
											: "Generate API Key"}
										<ChevronRight className="ml-2 h-4 w-4" />
									</Button>
								</div>
							</div>
						</CardContent>
					</Card>
				)}

				{/* Quickstart Step */}
				{currentStep === "quickstart" && createdApiKey && (
					<Card className="border-2">
						<CardHeader>
							<div className="mb-4 flex items-center gap-2">
								<Button
									variant="ghost"
									size="sm"
									onClick={handleBack}
									className="p-2"
								>
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
										Please save this API key somewhere safe. You won't be able
										to view it again.
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
										<code className="break-all font-mono text-sm">
											{createdApiKey}
										</code>
									</div>
								</div>
							</div>

							<Separator />

							{/* Quick Examples */}
							<div className="space-y-4">
								<div>
									<h3 className="font-semibold text-lg">🚀 Test Your API</h3>
									<p className="text-muted-foreground text-sm">
										Try these examples to get started
									</p>
								</div>

								<Tabs defaultValue="curl" className="w-full">
									<TabsList className="grid w-full grid-cols-3">
										<TabsTrigger value="curl">cURL</TabsTrigger>
										<TabsTrigger value="javascript">JavaScript</TabsTrigger>
										<TabsTrigger value="python">Python</TabsTrigger>
									</TabsList>

									<TabsContent value="curl" className="mt-4">
										<CodeBlock>
											<CodeBlockGroup className="border-b px-4 py-2">
												<span className="font-medium text-sm">
													Test with cURL
												</span>
												<div className="flex items-center gap-2">
													<Badge variant="secondary" className="text-xs">
														bash
													</Badge>
													<CopyButton
														content={`curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${createdApiKey}" \\
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 150
  }'`}
														copyMessage="cURL command copied!"
													/>
												</div>
											</CodeBlockGroup>
											<CodeBlockCode
												code={`curl -X POST "${API_BASE_URL}/api/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${createdApiKey}" \\
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 150
  }'`}
												language="bash"
											/>
										</CodeBlock>
									</TabsContent>

									<TabsContent value="javascript" className="mt-4">
										<div className="space-y-4">
											<div className="rounded-lg border bg-blue-50 p-3 dark:bg-blue-950/20">
												<p className="font-medium text-blue-900 text-sm dark:text-blue-100">
													Install:{" "}
													<code className="text-blue-700 dark:text-blue-300">
														npm install openai
													</code>
												</p>
											</div>
											<CodeBlock>
												<CodeBlockGroup className="border-b px-4 py-2">
													<span className="font-medium text-sm">
														JavaScript
													</span>
													<div className="flex items-center gap-2">
														<Badge variant="secondary" className="text-xs">
															javascript
														</Badge>
														<CopyButton
															content={`import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: '${createdApiKey}',
  baseURL: '${API_BASE_URL}/api/v1',
});

const completion = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 150,
});

console.log(completion.choices[0]);`}
															copyMessage="JavaScript code copied!"
														/>
													</div>
												</CodeBlockGroup>
												<CodeBlockCode
													code={`import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: '${createdApiKey}',
  baseURL: '${API_BASE_URL}/api/v1',
});

const completion = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 150,
});

console.log(completion.choices[0]);`}
													language="javascript"
												/>
											</CodeBlock>
										</div>
									</TabsContent>

									<TabsContent value="python" className="mt-4">
										<div className="space-y-4">
											<div className="rounded-lg border bg-blue-50 p-3 dark:bg-blue-950/20">
												<p className="font-medium text-blue-900 text-sm dark:text-blue-100">
													Install:{" "}
													<code className="text-blue-700 dark:text-blue-300">
														pip install openai
													</code>
												</p>
											</div>
											<CodeBlock>
												<CodeBlockGroup className="border-b px-4 py-2">
													<span className="font-medium text-sm">Python</span>
													<div className="flex items-center gap-2">
														<Badge variant="secondary" className="text-xs">
															python
														</Badge>
														<CopyButton
															content={`from openai import OpenAI

client = OpenAI(
    api_key="${createdApiKey}",
    base_url="${API_BASE_URL}/api/v1"
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=150
)

print(completion.choices[0].message.content)`}
															copyMessage="Python code copied!"
														/>
													</div>
												</CodeBlockGroup>
												<CodeBlockCode
													code={`from openai import OpenAI

client = OpenAI(
    api_key="${createdApiKey}",
    base_url="${API_BASE_URL}/api/v1"
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=150
)

print(completion.choices[0].message.content)`}
													language="python"
												/>
											</CodeBlock>
										</div>
									</TabsContent>
								</Tabs>
							</div>

							<div className="flex justify-end gap-3">
								<Button type="button" variant="outline" onClick={handleBack}>
									Back
								</Button>
								<Button
									onClick={() => setCurrentStep("complete")}
									className="min-w-32"
								>
									Continue to Dashboard
									<ChevronRight className="ml-2 h-4 w-4" />
								</Button>
							</div>
						</CardContent>
					</Card>
				)}

				{/* Complete Step */}
				{currentStep === "complete" && (
					<Card className="border-2">
						<CardHeader className="text-center">
							<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-success/10">
								<CheckCircle className="h-8 w-8 text-success" />
							</div>
							<CardTitle className="text-2xl">You're all set!</CardTitle>
							<CardDescription className="text-base">
								Your organization and project have been created successfully.
								You can now start using the Adaptive AI Platform.
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
								<Button onClick={handleComplete} className="w-full" size="lg">
									Go to Dashboard
									<ChevronRight className="ml-2 h-4 w-4" />
								</Button>
							</div>
						</CardContent>
					</Card>
				)}
			</div>
		</div>
	);
}
