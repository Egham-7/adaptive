// src/server/api/routers/conversation.ts

import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
	createConversationSchema,
	getConversationsOptionsSchema,
	updateConversationSchema,
} from "@/lib/chat/schema";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

export const conversationRouter = createTRPCRouter({
	create: protectedProcedure
		.input(createConversationSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			return ctx.db.conversation.create({
				data: {
					...input,
					userId: userId,
					pinned: input.pinned ?? false,
				},
			});
		}),

	getById: protectedProcedure
		.input(z.object({ id: z.number() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			const conversation = await ctx.db.conversation.findUnique({
				where: { id: input.id, userId: userId, deletedAt: null },
				include: {
					messages: {
						where: { deletedAt: null },
						orderBy: { createdAt: "asc" },
					},
				},
			});

			if (!conversation) {
				throw new TRPCError({
					code: "NOT_FOUND",
					message: "Conversation not found or you do not have access.",
				});
			}
			return conversation;
		}),

	list: protectedProcedure
		.input(getConversationsOptionsSchema.optional())
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			return ctx.db.conversation.findMany({
				where: {
					userId: userId,
					deletedAt: null,
					...(input?.pinned !== undefined && { pinned: input.pinned }),
				},
				// We include the last message to display in the sidebar
				include: {
					messages: {
						where: { deletedAt: null },
						orderBy: { updatedAt: "desc" },
						take: 1,
					},
				},
				orderBy: [{ pinned: "desc" }, { updatedAt: "desc" }],
			});
		}),

	update: protectedProcedure
		.input(updateConversationSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			const { id, ...dataToUpdate } = input;

			return ctx.db.$transaction(async (tx) => {
				const existingConversation = await tx.conversation.findFirst({
					where: { id: id, userId: userId, deletedAt: null },
				});

				if (!existingConversation) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message:
							"Conversation not found or you do not have access to update it.",
					});
				}

				return tx.conversation.update({
					where: { id: id },
					data: {
						...dataToUpdate,
						updatedAt: new Date(),
					},
				});
			});
		}),

	delete: protectedProcedure
		.input(z.object({ id: z.number() }))
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			return ctx.db.$transaction(async (tx) => {
				const conversationToDelete = await tx.conversation.findFirst({
					where: { id: input.id, userId: userId, deletedAt: null },
				});

				if (!conversationToDelete) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message:
							"Conversation not found or you do not have access to delete it.",
					});
				}

				const deletedConversation = await tx.conversation.update({
					where: { id: input.id },
					data: { deletedAt: new Date() },
				});

				await tx.message.updateMany({
					where: { conversationId: input.id, deletedAt: null },
					data: { deletedAt: new Date() },
				});

				return deletedConversation;
			});
		}),

	setPinStatus: protectedProcedure
		.input(z.object({ id: z.number(), pinned: z.boolean() }))
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			const { id, pinned } = input;

			return ctx.db.$transaction(async (tx) => {
				const existingConversation = await tx.conversation.findFirst({
					where: { id, userId, deletedAt: null },
				});

				if (!existingConversation) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Conversation not found.",
					});
				}

				return tx.conversation.update({
					where: { id },
					data: { pinned, updatedAt: new Date() },
				});
			});
		}),
});
