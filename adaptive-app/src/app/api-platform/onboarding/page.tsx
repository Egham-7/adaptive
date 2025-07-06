"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import {
	ArrowLeft,
	Building2,
	CheckCircle,
	ChevronRight,
	FolderPlus,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
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
import { Textarea } from "@/components/ui/textarea";
import { useCreateOrganization } from "@/hooks/organizations/use-create-organization";
import { useCreateProject } from "@/hooks/projects/use-create-project";
import type { OrganizationCreateResponse } from "@/types";

const organizationSchema = z.object({
	name: z.string().min(1, "Organization name is required"),
	description: z.string().optional(),
});

const projectSchema = z.object({
	name: z.string().min(1, "Project name is required"),
	description: z.string().optional(),
});

type OnboardingStep = "welcome" | "organization" | "project" | "complete";

export default function OnboardingPage() {
	const [currentStep, setCurrentStep] = useState<OnboardingStep>("welcome");
	const [createdOrganization, setCreatedOrganization] =
		useState<OrganizationCreateResponse | null>(null);
	const router = useRouter();

	const createOrganization = useCreateOrganization();
	const createProject = useCreateProject();

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
				onSuccess: () => {
					setCurrentStep("complete");
				},

				onError: (error) => {
					console.error("Failed to create project: ", error);
					toast.error(`Failed to create project: ${error.message}`);
				},
			},
		);
	};

	const getStepNumber = (step: OnboardingStep): number => {
		switch (step) {
			case "welcome":
				return 1;
			case "organization":
				return 2;
			case "project":
				return 3;
			case "complete":
				return 4;
			default:
				return 1;
		}
	};

	const getProgress = (): number => {
		return ((getStepNumber(currentStep) - 1) / 3) * 100;
	};

	const handleComplete = () => {
		if (createdOrganization) {
			router.push(`/api-platform/organizations/${createdOrganization.id}`);
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
						<span>Step {getStepNumber(currentStep)} of 4</span>
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
