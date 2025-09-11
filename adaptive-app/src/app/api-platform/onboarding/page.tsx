"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";
import { ApiKeyStep } from "@/app/_components/api-platform/onboarding/api-key-step";
import { CompleteStep } from "@/app/_components/api-platform/onboarding/complete-step";
import { OrganizationStep } from "@/app/_components/api-platform/onboarding/organization-step";
import { ProjectStep } from "@/app/_components/api-platform/onboarding/project-step";
import { QuickstartStep } from "@/app/_components/api-platform/onboarding/quickstart-step";
import { WelcomeStep } from "@/app/_components/api-platform/onboarding/welcome-step";
import { Progress } from "@/components/ui/progress";
import { useCreateProjectApiKey } from "@/hooks/api_keys/use-create-project-api-key";
import { useCreateOrganization } from "@/hooks/organizations/use-create-organization";
import { useCreateProject } from "@/hooks/projects/use-create-project";
import { setLastProject } from "@/hooks/use-smart-redirect";
import { api } from "@/trpc/react";
import type {
	OrganizationCreateResponse,
	ProjectCreateResponse,
} from "@/types";

type OnboardingStep =
	| "welcome"
	| "organization"
	| "project"
	| "api-key"
	| "quickstart"
	| "complete";

export default function OnboardingPage() {
	const [currentStep, setCurrentStep] = useState<OnboardingStep>("welcome");
	const [createdOrganization, setCreatedOrganization] =
		useState<OrganizationCreateResponse | null>(null);

	const [createdProject, setCreatedProject] =
		useState<ProjectCreateResponse | null>(null);
	const [createdApiKey, setCreatedApiKey] = useState<string | null>(null);
	const router = useRouter();

	const createOrganization = useCreateOrganization();
	const createProject = useCreateProject();
	const createApiKey = useCreateProjectApiKey();
	const revealApiKey = api.api_keys.revealApiKey.useMutation();

	const onOrganizationSubmit = (values: {
		name: string;
		description?: string;
	}) => {
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

	const onProjectSubmit = (values: { name: string; description?: string }) => {
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
					<WelcomeStep onContinue={() => setCurrentStep("organization")} />
				)}

				{/* Organization Step */}
				{currentStep === "organization" && (
					<OrganizationStep
						onSubmit={onOrganizationSubmit}
						onBack={handleBack}
						isLoading={createOrganization.isPending}
					/>
				)}

				{/* Project Step */}
				{currentStep === "project" && (
					<ProjectStep
						onSubmit={onProjectSubmit}
						onBack={handleBack}
						isLoading={createProject.isPending}
					/>
				)}

				{/* API Key Step */}
				{currentStep === "api-key" && (
					<ApiKeyStep
						onCreateApiKey={handleCreateApiKey}
						onBack={handleBack}
						isLoading={createApiKey.isPending}
					/>
				)}

				{/* Quickstart Step */}
				{currentStep === "quickstart" && createdApiKey && (
					<QuickstartStep
						apiKey={createdApiKey}
						onContinue={() => setCurrentStep("complete")}
						onBack={handleBack}
					/>
				)}

				{/* Complete Step */}
				{currentStep === "complete" && (
					<CompleteStep onComplete={handleComplete} />
				)}
			</div>
		</div>
	);
}
