"use client";

import { useUser } from "@clerk/nextjs";
import { zodResolver } from "@hookform/resolvers/zod";
import {
	Activity,
	ArrowLeft,
	Calendar,
	CheckCircle,
	ChevronRight,
	Clock,
	Edit,
	Folder,
	Pause,
	Plus,
	Search,
	Trash2,
	Users,
	XCircle,
} from "lucide-react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { CreditManagement } from "@/app/_components/api-platform/organizations/credit-management";
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
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useOrganization } from "@/hooks/organizations/use-organization";
import { useCreateProject } from "@/hooks/projects/use-create-project";
import { useDeleteProject } from "@/hooks/projects/use-delete-project";
import { useProjects } from "@/hooks/projects/use-projects";
import { useUpdateProject } from "@/hooks/projects/use-update-project";
import { setLastOrganization } from "@/hooks/use-smart-redirect";
import type { OrganizationDetails, ProjectListItem } from "@/types";

const createProjectSchema = z.object({
	name: z.string().min(1, "Project name is required"),
	description: z.string().optional(),
	status: z.enum(["active", "inactive", "paused"]),
});

const updateProjectSchema = z.object({
	name: z.string().min(1, "Project name is required"),
	description: z.string().optional(),
	status: z.enum(["active", "inactive", "paused"]),
});

export default function OrganizationProjectsPage() {
	const params = useParams();
	const searchParams = useSearchParams();
	const orgId = params.orgId as string;
	const [searchQuery, setSearchQuery] = useState("");
	const [showCreateDialog, setShowCreateDialog] = useState(false);
	const [editingProject, setEditingProject] = useState<ProjectListItem | null>(
		null,
	);

	// Check if we're showing credits tab with validation
	const tab = searchParams.get("tab");
	const validTabs = ["projects", "credits"] as const;
	const showCredits = tab === "credits";

	// Validate tab parameter with proper typing
	const isValidTab = !tab || (validTabs as readonly string[]).includes(tab);

	const { user } = useUser();
	const {
		data: organization,
		isLoading: orgLoading,
		error: orgError,
	} = useOrganization(orgId);
	const { data: projects = [], isLoading: projectsLoading } =
		useProjects(orgId);
	const createProject = useCreateProject();
	const updateProject = useUpdateProject();
	const deleteProject = useDeleteProject();

	// Track organization visit for smart redirect
	useEffect(() => {
		if (orgId) {
			setLastOrganization(orgId);
		}
	}, [orgId]);

	const createForm = useForm<z.infer<typeof createProjectSchema>>({
		resolver: zodResolver(createProjectSchema),
		defaultValues: {
			name: "",
			description: "",
			status: "active",
		},
	});

	const editForm = useForm<z.infer<typeof updateProjectSchema>>({
		resolver: zodResolver(updateProjectSchema),
		defaultValues: {
			name: "",
			description: "",
			status: "active",
		},
	});

	const filteredProjects = projects.filter(
		(project) =>
			project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
			project.description?.toLowerCase().includes(searchQuery.toLowerCase()),
	);

	const getStatusIcon = (status: string) => {
		switch (status) {
			case "active":
				return <CheckCircle className="h-4 w-4 text-green-600" />;
			case "inactive":
				return <XCircle className="h-4 w-4 text-destructive" />;
			case "paused":
				return <Pause className="h-4 w-4 text-amber-600" />;
			default:
				return <Clock className="h-4 w-4 text-muted-foreground" />;
		}
	};

	const getStatusBadge = (status: string) => {
		switch (status) {
			case "active":
				return (
					<Badge
						variant="outline"
						className="border-success/20 bg-success/10 text-success"
					>
						Active
					</Badge>
				);
			case "inactive":
				return <Badge variant="destructive">Inactive</Badge>;
			case "paused":
				return (
					<Badge
						variant="outline"
						className="border-warning/20 bg-warning/10 text-warning"
					>
						Paused
					</Badge>
				);
			default:
				return <Badge variant="outline">Unknown</Badge>;
		}
	};

	const getUserRole = (org: OrganizationDetails, userId: string) => {
		if (org?.ownerId === userId) return "Owner";
		const member = org?.members?.find((m) => m.userId === userId);
		return member?.role || "Member";
	};

	const onCreateSubmit = (values: z.infer<typeof createProjectSchema>) => {
		createProject.mutate(
			{
				...values,
				organizationId: orgId,
			},
			{
				onSuccess: () => {
					setShowCreateDialog(false);
					createForm.reset();
				},
			},
		);
	};

	const onEditSubmit = (values: z.infer<typeof updateProjectSchema>) => {
		if (!editingProject) return;

		updateProject.mutate(
			{
				id: editingProject.id,
				...values,
			},
			{
				onSuccess: () => {
					setEditingProject(null);
					editForm.reset();
				},
			},
		);
	};

	const handleEditProject = (project: ProjectListItem, e: React.MouseEvent) => {
		e.preventDefault();
		e.stopPropagation();
		setEditingProject(project);
		editForm.reset({
			name: project.name,
			description: project.description || "",
			status: project.status as "active" | "inactive" | "paused",
		});
	};

	const handleDeleteProject = (id: string, e: React.MouseEvent) => {
		e.preventDefault();
		e.stopPropagation();
		deleteProject.mutate({ id });
	};

	if (orgLoading || projectsLoading) {
		return (
			<div className="min-h-screen bg-background">
				<div className="container mx-auto max-w-6xl px-4 py-8">
					<div className="flex min-h-[400px] items-center justify-center">
						<div className="text-center">
							<div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
							<p className="text-muted-foreground">Loading organization...</p>
						</div>
					</div>
				</div>
			</div>
		);
	}

	if (orgError || !organization) {
		return (
			<div className="min-h-screen bg-background">
				<div className="container mx-auto max-w-6xl px-4 py-8">
					<div className="flex min-h-[400px] items-center justify-center">
						<div className="text-center">
							<h3 className="mb-2 font-medium text-foreground text-lg">
								Organization not found
							</h3>
							<p className="mb-4 text-muted-foreground">
								{orgError?.message ||
									"The organization you're looking for doesn't exist or you don't have access to it."}
							</p>
							<Link href="/api-platform/organizations">
								<Button>Back to Organizations</Button>
							</Link>
						</div>
					</div>
				</div>
			</div>
		);
	}

	const userRole = getUserRole(organization, user?.id || "");
	const canCreateProject = userRole === "Owner" || userRole === "admin";
	const canEditProject = userRole === "Owner" || userRole === "admin";
	const canDeleteProject = userRole === "Owner";

	return (
		<div className="w-full">
			{/* Back Navigation */}
			<div className="mb-6">
				<Link href="/api-platform/organizations">
					<Button variant="ghost" size="sm">
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Organizations
					</Button>
				</Link>
			</div>

			{/* Header */}
			<div className="mb-8">
				<h1 className="mb-2 font-bold text-3xl text-foreground">
					{organization.name}
				</h1>
				<p className="mb-6 text-muted-foreground">{organization.description}</p>

				{/* Stats */}
				<div className="flex flex-wrap gap-8">
					<div className="flex items-center gap-2">
						<div className="rounded-lg bg-muted p-2">
							<Folder className="h-5 w-5 text-muted-foreground" />
						</div>
						<div>
							<div className="font-semibold text-foreground">
								{projects.length}
							</div>
							<div className="text-muted-foreground text-sm">Projects</div>
						</div>
					</div>
					<div className="flex items-center gap-2">
						<div className="rounded-lg bg-muted p-2">
							<Activity className="h-5 w-5 text-muted-foreground" />
						</div>
						<div>
							<div className="font-semibold text-foreground">
								{projects.filter((p) => p.status === "active").length}
							</div>
							<div className="text-muted-foreground text-sm">Active</div>
						</div>
					</div>
				</div>
			</div>

			{/* Show Credits or Projects based on URL */}
			{!isValidTab ? (
				<div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950/20">
					<div className="flex items-center gap-2 text-red-700 dark:text-red-400">
						<span className="font-medium text-sm">Invalid Tab</span>
					</div>
					<p className="mt-2 text-red-600 text-sm dark:text-red-300">
						The tab "{tab}" is not valid. Available tabs: projects, credits
					</p>
				</div>
			) : showCredits ? (
				<CreditManagement organizationId={orgId} />
			) : (
				<>
					{/* Search */}
					<div className="mb-8">
						<div className="relative max-w-md">
							<Search className="-translate-y-1/2 absolute top-1/2 left-3 h-4 w-4 transform text-muted-foreground" />
							<Input
								placeholder="Search projects..."
								value={searchQuery}
								onChange={(e) => setSearchQuery(e.target.value)}
								className="pl-10"
							/>
						</div>
					</div>

					{/* Projects Grid */}
					<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
						{/* Create Project Card */}
						{canCreateProject && (
							<Dialog
								open={showCreateDialog}
								onOpenChange={setShowCreateDialog}
							>
								<DialogTrigger asChild>
									<Card className="cursor-pointer border-2 border-muted-foreground/25 border-dashed transition-shadow hover:border-muted-foreground/50 hover:shadow-md">
										<CardContent className="flex flex-col items-center justify-center py-12">
											<div className="mb-4 rounded-lg bg-muted p-4">
												<Plus className="h-8 w-8 text-muted-foreground" />
											</div>
											<CardTitle className="mb-2 text-lg">
												Create Project
											</CardTitle>
											<CardDescription className="text-center">
												Start a new project to organize your work and
												collaborate with your team
											</CardDescription>
										</CardContent>
									</Card>
								</DialogTrigger>
								<DialogContent className="max-w-md">
									<DialogHeader>
										<DialogTitle>Create New Project</DialogTitle>
									</DialogHeader>
									<Form {...createForm}>
										<form
											onSubmit={createForm.handleSubmit(onCreateSubmit)}
											className="space-y-4"
										>
											<FormField
												control={createForm.control}
												name="name"
												render={({ field }) => (
													<FormItem>
														<FormLabel>Project Name</FormLabel>
														<FormControl>
															<Input
																placeholder="Enter project name"
																{...field}
															/>
														</FormControl>
														<FormMessage />
													</FormItem>
												)}
											/>
											<FormField
												control={createForm.control}
												name="description"
												render={({ field }) => (
													<FormItem>
														<FormLabel>Description (optional)</FormLabel>
														<FormControl>
															<Textarea
																placeholder="What's this project about?"
																rows={3}
																{...field}
															/>
														</FormControl>
														<FormMessage />
													</FormItem>
												)}
											/>
											<FormField
												control={createForm.control}
												name="status"
												render={({ field }) => (
													<FormItem>
														<FormLabel>Status</FormLabel>
														<Select
															onValueChange={field.onChange}
															defaultValue={field.value}
														>
															<FormControl>
																<SelectTrigger>
																	<SelectValue placeholder="Select status" />
																</SelectTrigger>
															</FormControl>
															<SelectContent>
																<SelectItem value="active">Active</SelectItem>
																<SelectItem value="inactive">
																	Inactive
																</SelectItem>
																<SelectItem value="paused">Paused</SelectItem>
															</SelectContent>
														</Select>
														<FormMessage />
													</FormItem>
												)}
											/>
											<div className="flex justify-end gap-2">
												<Button
													type="button"
													variant="outline"
													onClick={() => {
														setShowCreateDialog(false);
														createForm.reset();
													}}
												>
													Cancel
												</Button>
												<Button
													type="submit"
													disabled={createProject.isPending}
												>
													{createProject.isPending
														? "Creating..."
														: "Create Project"}
												</Button>
											</div>
										</form>
									</Form>
								</DialogContent>
							</Dialog>
						)}

						{filteredProjects.map((project) => {
							const canEdit = canEditProject;
							const canDelete = canDeleteProject && projects.length > 1;

							return (
								<Card
									key={project.id}
									className="group relative transition-shadow hover:shadow-md"
								>
									{/* Edit/Delete Actions */}
									{canEdit && (
										<div className="absolute top-4 right-4 z-10 opacity-0 transition-opacity group-hover:opacity-100">
											<div className="flex gap-1">
												<Button
													variant="ghost"
													size="sm"
													onClick={(e) => handleEditProject(project, e)}
													className="h-8 w-8 p-0 hover:bg-muted"
												>
													<Edit className="h-4 w-4" />
												</Button>
												{canDelete && (
													<Button
														variant="ghost"
														size="sm"
														onClick={(e) => handleDeleteProject(project.id, e)}
														className="h-8 w-8 p-0 hover:bg-destructive hover:text-destructive-foreground"
													>
														<Trash2 className="h-4 w-4" />
													</Button>
												)}
											</div>
										</div>
									)}

									<Link
										href={`/api-platform/organizations/${orgId}/projects/${project.id}`}
									>
										<CardHeader className="pb-3">
											<div className="mb-3 flex items-start justify-between">
												<div className="flex items-center gap-3">
													<div className="rounded-lg bg-muted p-2">
														<Folder className="h-5 w-5 text-muted-foreground" />
													</div>
													<div className="min-w-0 flex-1">
														<CardTitle className="truncate text-lg">
															{project.name}
														</CardTitle>
													</div>
												</div>
												<ChevronRight className="h-4 w-4 text-muted-foreground" />
											</div>

											<div className="mb-3 flex items-center gap-2">
												{getStatusIcon(project.status)}
												{getStatusBadge(project.status)}
											</div>
										</CardHeader>

										<CardContent className="pt-0">
											<CardDescription className="mb-4 line-clamp-2 text-muted-foreground leading-relaxed">
												{project.description}
											</CardDescription>

											<div className="space-y-3 text-sm">
												<div className="flex items-center justify-between">
													<div className="flex items-center gap-2 text-muted-foreground">
														<Users className="h-4 w-4" />
														<span>{project.members?.length || 0} members</span>
													</div>
													<div className="flex items-center gap-2 text-muted-foreground">
														<Calendar className="h-4 w-4" />
														<span>
															{new Date(project.createdAt).toLocaleDateString()}
														</span>
													</div>
												</div>
												<div className="flex items-center gap-2 text-muted-foreground">
													<Clock className="h-4 w-4" />
													<span>
														Updated{" "}
														{new Date(project.updatedAt).toLocaleDateString()}
													</span>
												</div>
											</div>
										</CardContent>
									</Link>
								</Card>
							);
						})}
					</div>

					{filteredProjects.length === 0 && (
						<div className="py-12 text-center">
							<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-muted p-4">
								<Folder className="h-8 w-8 text-muted-foreground" />
							</div>
							<h3 className="mb-2 font-semibold text-foreground text-lg">
								No projects found
							</h3>
							<p className="mx-auto max-w-md text-muted-foreground">
								{searchQuery
									? "Try adjusting your search terms"
									: "This organization doesn't have any projects yet"}
							</p>
						</div>
					)}
				</>
			)}

			{/* Edit Project Dialog */}
			<Dialog
				open={!!editingProject}
				onOpenChange={(open) => !open && setEditingProject(null)}
			>
				<DialogContent className="max-w-md">
					<DialogHeader>
						<DialogTitle>Edit Project</DialogTitle>
					</DialogHeader>
					<Form {...editForm}>
						<form
							onSubmit={editForm.handleSubmit(onEditSubmit)}
							className="space-y-4"
						>
							<FormField
								control={editForm.control}
								name="name"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Project Name</FormLabel>
										<FormControl>
											<Input placeholder="Enter project name" {...field} />
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>
							<FormField
								control={editForm.control}
								name="description"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Description (optional)</FormLabel>
										<FormControl>
											<Textarea
												placeholder="What's this project about?"
												rows={3}
												{...field}
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>
							<FormField
								control={editForm.control}
								name="status"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Status</FormLabel>
										<Select
											onValueChange={field.onChange}
											defaultValue={field.value}
										>
											<FormControl>
												<SelectTrigger>
													<SelectValue placeholder="Select status" />
												</SelectTrigger>
											</FormControl>
											<SelectContent>
												<SelectItem value="active">Active</SelectItem>
												<SelectItem value="inactive">Inactive</SelectItem>
												<SelectItem value="paused">Paused</SelectItem>
											</SelectContent>
										</Select>
										<FormMessage />
									</FormItem>
								)}
							/>
							<div className="flex justify-end gap-2">
								<Button
									type="button"
									variant="outline"
									onClick={() => {
										setEditingProject(null);
										editForm.reset();
									}}
								>
									Cancel
								</Button>
								<Button type="submit" disabled={updateProject.isPending}>
									{updateProject.isPending ? "Updating..." : "Update Project"}
								</Button>
							</div>
						</form>
					</Form>
				</DialogContent>
			</Dialog>
		</div>
	);
}
