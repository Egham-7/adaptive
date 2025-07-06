"use client";

import { useUser } from "@clerk/nextjs";
import { zodResolver } from "@hookform/resolvers/zod";
import {
	Building2,
	ChevronRight,
	Edit,
	Plus,
	Search,
	Trash2,
	Users,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
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
import { Textarea } from "@/components/ui/textarea";
import { useCreateOrganization } from "@/hooks/organizations/use-create-organization";
import { useDeleteOrganization } from "@/hooks/organizations/use-delete-organization";
import { useOrganizations } from "@/hooks/organizations/use-organizations";
import { useUpdateOrganization } from "@/hooks/organizations/use-update-organization";
import type { OrganizationListItem } from "@/types";

const createOrganizationSchema = z.object({
	name: z.string().min(1, "Organization name is required"),
	description: z.string().optional(),
});

const updateOrganizationSchema = z.object({
	name: z.string().min(1, "Organization name is required"),
	description: z.string().optional(),
});

export default function OrganizationsPage() {
	const [searchQuery, setSearchQuery] = useState("");
	const [showCreateDialog, setShowCreateDialog] = useState(false);
	const [editingOrganization, setEditingOrganization] =
		useState<OrganizationListItem | null>(null);

	const { user } = useUser();
	const router = useRouter();
	const { data: organizations = [], isLoading, error } = useOrganizations();
	const createOrganization = useCreateOrganization();
	const updateOrganization = useUpdateOrganization();
	const deleteOrganization = useDeleteOrganization();

	// Redirect to onboarding if user has no organizations
	useEffect(() => {
		if (!isLoading && organizations.length === 0 && !error) {
			router.push("/api-platform/onboarding");
		}
	}, [isLoading, organizations.length, error, router]);

	const createForm = useForm<z.infer<typeof createOrganizationSchema>>({
		resolver: zodResolver(createOrganizationSchema),
		defaultValues: {
			name: "",
			description: "",
		},
	});

	const editForm = useForm<z.infer<typeof updateOrganizationSchema>>({
		resolver: zodResolver(updateOrganizationSchema),
		defaultValues: {
			name: "",
			description: "",
		},
	});

	const filteredOrganizations = organizations.filter(
		(org) =>
			org.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
			org.description?.toLowerCase().includes(searchQuery.toLowerCase()),
	);

	const getUserRole = (org: OrganizationListItem, userId: string) => {
		if (org.ownerId === userId) return "Owner";
		const member = org.members?.find((m) => m.userId === userId);
		return member?.role || "Member";
	};

	const getRoleBadgeVariant = (role: string) => {
		switch (role.toLowerCase()) {
			case "owner":
				return "default";
			case "admin":
				return "secondary";
			case "member":
				return "outline";
			default:
				return "outline";
		}
	};

	const onCreateSubmit = (values: z.infer<typeof createOrganizationSchema>) => {
		createOrganization.mutate(values, {
			onSuccess: () => {
				setShowCreateDialog(false);
				createForm.reset();
			},
		});
	};

	const onEditSubmit = (values: z.infer<typeof updateOrganizationSchema>) => {
		if (!editingOrganization) return;

		updateOrganization.mutate(
			{
				id: editingOrganization.id,
				...values,
			},
			{
				onSuccess: () => {
					setEditingOrganization(null);
					editForm.reset();
				},
			},
		);
	};

	const handleDeleteOrganization = (id: string, e: React.MouseEvent) => {
		e.preventDefault();
		e.stopPropagation();
		deleteOrganization.mutate({ id });
	};

	const handleEditOrganization = (
		organization: OrganizationListItem,
		e: React.MouseEvent,
	) => {
		e.preventDefault();
		e.stopPropagation();
		setEditingOrganization(organization);
		editForm.reset({
			name: organization.name,
			description: organization.description || "",
		});
	};

	// Show loading while checking organizations or redirecting
	if (isLoading || (organizations.length === 0 && !error)) {
		return (
			<div className="min-h-screen bg-background">
				<div className="container mx-auto max-w-6xl px-4 py-8">
					<div className="mb-8">
						<h1 className="mb-2 font-bold text-3xl text-foreground">
							Organizations
						</h1>
						<p className="text-muted-foreground">
							Manage your AI projects and collaborate with teams
						</p>
					</div>
					<div className="flex min-h-[400px] items-center justify-center">
						<div className="text-center">
							<div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
							<p className="text-muted-foreground">
								{organizations.length === 0 && !error
									? "Setting up your workspace..."
									: "Loading organizations..."}
							</p>
						</div>
					</div>
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="min-h-screen bg-background">
				<div className="container mx-auto max-w-6xl px-4 py-8">
					<div className="mb-8">
						<h1 className="mb-2 font-bold text-3xl text-foreground">
							Organizations
						</h1>
						<p className="text-muted-foreground">
							Manage your AI projects and collaborate with teams
						</p>
					</div>
					<div className="flex min-h-[400px] items-center justify-center">
						<div className="text-center">
							<h3 className="mb-2 font-medium text-foreground text-lg">
								Failed to load organizations
							</h3>
							<p className="mb-4 text-muted-foreground">{error.message}</p>
						</div>
					</div>
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-background">
			<div className="container mx-auto max-w-6xl px-4 py-8">
				<div className="mb-8">
					<h1 className="mb-2 font-bold text-3xl text-foreground">
						Organizations
					</h1>
					<p className="text-muted-foreground">
						Manage your AI projects and collaborate with teams
					</p>
				</div>

				<div className="mb-8">
					<div className="relative max-w-md">
						<Search className="-translate-y-1/2 absolute top-1/2 left-3 h-4 w-4 transform text-muted-foreground" />
						<Input
							placeholder="Search organizations..."
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							className="pl-10"
						/>
					</div>
				</div>

				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
					{/* Create Organization Card */}
					<Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
						<DialogTrigger asChild>
							<Card className="cursor-pointer border-2 border-muted-foreground/25 border-dashed transition-shadow hover:border-muted-foreground/50 hover:shadow-md">
								<CardContent className="flex flex-col items-center justify-center py-12">
									<div className="mb-4 rounded-lg bg-muted p-4">
										<Plus className="h-8 w-8 text-muted-foreground" />
									</div>
									<CardTitle className="mb-2 text-lg">
										Create Organization
									</CardTitle>
									<CardDescription className="text-center">
										Start a new organization to manage your AI projects and
										collaborate with teams
									</CardDescription>
								</CardContent>
							</Card>
						</DialogTrigger>
						<DialogContent className="max-w-md">
							<DialogHeader>
								<DialogTitle>Create New Organization</DialogTitle>
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
												<FormLabel>Organization Name</FormLabel>
												<FormControl>
													<Input
														placeholder="Enter organization name"
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
														placeholder="What's this organization for?"
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
											disabled={createOrganization.isPending}
										>
											{createOrganization.isPending
												? "Creating..."
												: "Create Organization"}
										</Button>
									</div>
								</form>
							</Form>
						</DialogContent>
					</Dialog>

					{filteredOrganizations.map((org) => {
						const userRole = getUserRole(org, user?.id || "");
						const canEdit = userRole === "Owner" || userRole === "Admin";
						const canDelete = userRole === "Owner" && organizations.length > 1;

						return (
							<Card
								key={org.id}
								className="group relative transition-shadow hover:shadow-md"
							>
								{/* Edit/Delete Actions */}
								{canEdit && (
									<div className="absolute top-4 right-4 z-10 opacity-0 transition-opacity group-hover:opacity-100">
										<div className="flex gap-1">
											<Button
												variant="ghost"
												size="sm"
												onClick={(e) => handleEditOrganization(org, e)}
												className="h-8 w-8 p-0 hover:bg-muted"
											>
												<Edit className="h-4 w-4" />
											</Button>
											{canDelete && (
												<Button
													variant="ghost"
													size="sm"
													onClick={(e) => handleDeleteOrganization(org.id, e)}
													className="h-8 w-8 p-0 hover:bg-destructive hover:text-destructive-foreground"
												>
													<Trash2 className="h-4 w-4" />
												</Button>
											)}
										</div>
									</div>
								)}

								<Link href={`/api-platform/organizations/${org.id}`}>
									<CardHeader className="pb-4">
										<div className="flex items-start justify-between">
											<div className="flex items-center gap-3">
												<div className="rounded-lg bg-muted p-2">
													<Building2 className="h-5 w-5 text-muted-foreground" />
												</div>
												<div>
													<CardTitle className="text-lg">{org.name}</CardTitle>
													<Badge
														variant={getRoleBadgeVariant(userRole)}
														className="mt-1"
													>
														{userRole}
													</Badge>
												</div>
											</div>
											<ChevronRight className="h-5 w-5 text-muted-foreground" />
										</div>
									</CardHeader>
									<CardContent className="pt-0">
										<CardDescription className="mb-4 text-muted-foreground">
											{org.description || "No description provided"}
										</CardDescription>
										<div className="flex items-center gap-4 text-muted-foreground text-sm">
											<div className="flex items-center gap-1">
												<Users className="h-4 w-4" />
												<span>{org.members?.length || 0} members</span>
											</div>
											<div>
												<span>{org._count?.projects || 0} projects</span>
											</div>
										</div>
									</CardContent>
								</Link>
							</Card>
						);
					})}
				</div>

				{filteredOrganizations.length === 0 && (
					<div className="py-12 text-center">
						<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-muted p-4">
							<Building2 className="h-8 w-8 text-muted-foreground" />
						</div>
						<h3 className="mb-2 font-semibold text-foreground text-lg">
							No organizations found
						</h3>
						<p className="mx-auto max-w-md text-muted-foreground">
							{searchQuery
								? "Try adjusting your search terms"
								: "You don't belong to any organizations yet"}
						</p>
					</div>
				)}

				{/* Edit Organization Dialog */}
				<Dialog
					open={!!editingOrganization}
					onOpenChange={(open) => !open && setEditingOrganization(null)}
				>
					<DialogContent className="max-w-md">
						<DialogHeader>
							<DialogTitle>Edit Organization</DialogTitle>
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
											<FormLabel>Organization Name</FormLabel>
											<FormControl>
												<Input
													placeholder="Enter organization name"
													{...field}
												/>
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
													placeholder="What's this organization for?"
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
										type="button"
										variant="outline"
										onClick={() => {
											setEditingOrganization(null);
											editForm.reset();
										}}
									>
										Cancel
									</Button>
									<Button type="submit" disabled={updateOrganization.isPending}>
										{updateOrganization.isPending
											? "Updating..."
											: "Update Organization"}
									</Button>
								</div>
							</form>
						</Form>
					</DialogContent>
				</Dialog>
			</div>
		</div>
	);
}
