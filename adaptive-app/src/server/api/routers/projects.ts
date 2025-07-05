import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

const _projectSchema = z.object({
	id: z.string(),
	name: z.string(),
	description: z.string().nullable(),
	status: z.enum(["active", "inactive", "paused"]),
	progress: z.number(),
	organizationId: z.string(),
	createdAt: z.date(),
	updatedAt: z.date(),
});

const _projectMemberSchema = z.object({
	id: z.string(),
	userId: z.string(),
	projectId: z.string(),
	role: z.enum(["owner", "admin", "member"]),
	createdAt: z.date(),
	updatedAt: z.date(),
});

export const projectsRouter = createTRPCRouter({
	// Get all projects for an organization
	getByOrganization: protectedProcedure
		.input(z.object({ organizationId: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Check if user has access to the organization
				const organization = await ctx.db.organization.findFirst({
					where: {
						id: input.organizationId,
						OR: [{ ownerId: userId }, { members: { some: { userId } } }],
					},
				});

				if (!organization) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "You don't have access to this organization",
					});
				}

				const projects = await ctx.db.project.findMany({
					where: {
						organizationId: input.organizationId,
					},
					include: {
						members: true,
					},
					orderBy: { createdAt: "desc" },
				});

				return projects;
			} catch (error) {
				console.error("Error fetching projects:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch projects",
				});
			}
		}),

	// Get a specific project by ID
	getById: protectedProcedure
		.input(z.object({ id: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.id,
						organization: {
							OR: [{ ownerId: userId }, { members: { some: { userId } } }],
						},
					},
					include: {
						members: true,
						organization: true,
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Project not found",
					});
				}

				return project;
			} catch (error) {
				console.error("Error fetching project:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch project",
				});
			}
		}),

	// Create a new project
	create: protectedProcedure
		.input(
			z.object({
				name: z.string().min(1, "Project name is required"),
				description: z.string().optional(),
				organizationId: z.string(),
				status: z.enum(["active", "inactive", "paused"]).default("active"),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Check if user has permission to create projects in this organization
				const organization = await ctx.db.organization.findFirst({
					where: {
						id: input.organizationId,
						OR: [
							{ ownerId: userId },
							{
								members: { some: { userId, role: { in: ["owner", "admin"] } } },
							},
						],
					},
				});

				if (!organization) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message:
							"You don't have permission to create projects in this organization",
					});
				}

				const project = await ctx.db.project.create({
					data: {
						name: input.name,
						description: input.description,
						organizationId: input.organizationId,
						status: input.status,
						members: {
							create: {
								userId: userId,
								role: "owner",
							},
						},
					},
					include: {
						members: true,
						organization: true,
					},
				});

				return project;
			} catch (error) {
				console.error("Error creating project:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to create project",
				});
			}
		}),

	// Update a project
	update: protectedProcedure
		.input(
			z.object({
				id: z.string(),
				name: z.string().min(1, "Project name is required").optional(),
				description: z.string().optional(),
				status: z.enum(["active", "inactive", "paused"]).optional(),
				progress: z.number().min(0).max(100).optional(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Check if user has permission to update this project
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.id,
						OR: [
							{
								members: { some: { userId, role: { in: ["owner", "admin"] } } },
							},
							{ organization: { ownerId: userId } },
							{
								organization: {
									members: {
										some: { userId, role: { in: ["owner", "admin"] } },
									},
								},
							},
						],
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "You don't have permission to update this project",
					});
				}

				const updatedProject = await ctx.db.project.update({
					where: { id: input.id },
					data: {
						...(input.name && { name: input.name }),
						...(input.description !== undefined && {
							description: input.description,
						}),
						...(input.status && { status: input.status }),
						...(input.progress !== undefined && { progress: input.progress }),
					},
					include: {
						members: true,
						organization: true,
					},
				});

				return updatedProject;
			} catch (error) {
				console.error("Error updating project:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update project",
				});
			}
		}),

	// Delete a project
	delete: protectedProcedure
		.input(z.object({ id: z.string() }))
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Check if user has permission to delete this project
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.id,
						OR: [
							{ members: { some: { userId, role: "owner" } } },
							{ organization: { ownerId: userId } },
						],
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "You don't have permission to delete this project",
					});
				}

				await ctx.db.project.delete({
					where: { id: input.id },
				});

				return { success: true };
			} catch (error) {
				console.error("Error deleting project:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to delete project",
				});
			}
		}),

	// Add a member to a project
	addMember: protectedProcedure
		.input(
			z.object({
				projectId: z.string(),
				userId: z.string(),
				role: z.enum(["admin", "member"]).default("member"),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const currentUserId = ctx.clerkAuth.userId;
			if (!currentUserId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Check if current user has permission to add members
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.projectId,
						OR: [
							{
								members: {
									some: {
										userId: currentUserId,
										role: { in: ["owner", "admin"] },
									},
								},
							},
							{ organization: { ownerId: currentUserId } },
							{
								organization: {
									members: {
										some: {
											userId: currentUserId,
											role: { in: ["owner", "admin"] },
										},
									},
								},
							},
						],
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "You don't have permission to add members to this project",
					});
				}

				// Check if user is already a member
				const existingMember = await ctx.db.projectMember.findUnique({
					where: {
						userId_projectId: {
							userId: input.userId,
							projectId: input.projectId,
						},
					},
				});

				if (existingMember) {
					throw new TRPCError({
						code: "CONFLICT",
						message: "User is already a member of this project",
					});
				}

				const member = await ctx.db.projectMember.create({
					data: {
						userId: input.userId,
						projectId: input.projectId,
						role: input.role,
					},
				});

				return member;
			} catch (error) {
				console.error("Error adding project member:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to add project member",
				});
			}
		}),

	// Remove a member from a project
	removeMember: protectedProcedure
		.input(
			z.object({
				projectId: z.string(),
				userId: z.string(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const currentUserId = ctx.clerkAuth.userId;
			if (!currentUserId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Check if current user has permission to remove members
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.projectId,
						OR: [
							{
								members: {
									some: {
										userId: currentUserId,
										role: { in: ["owner", "admin"] },
									},
								},
							},
							{ organization: { ownerId: currentUserId } },
							{
								organization: {
									members: {
										some: {
											userId: currentUserId,
											role: { in: ["owner", "admin"] },
										},
									},
								},
							},
						],
					},
					include: {
						members: true,
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message:
							"You don't have permission to remove members from this project",
					});
				}

				// Prevent removing the project owner
				const memberToRemove = project.members.find(
					(m) => m.userId === input.userId,
				);
				if (memberToRemove?.role === "owner") {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "Cannot remove the project owner",
					});
				}

				await ctx.db.projectMember.delete({
					where: {
						userId_projectId: {
							userId: input.userId,
							projectId: input.projectId,
						},
					},
				});

				return { success: true };
			} catch (error) {
				console.error("Error removing project member:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to remove project member",
				});
			}
		}),
});
