import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
	getRemainingMessages,
	hasReachedDailyLimit,
} from "@/lib/chat/message-limits";
import {
	createMessageData,
	createMessageWithTimestampUpdate,
	findConversationByUserAndId,
	findMessageWithConversationAccess,
	softDeleteMessageWithTimestampUpdate,
	updateMessageData,
	updateMessageWithTimestampUpdate,
	validateConversationAccess,
	validateMessageAccess,
} from "@/lib/server/message-utils";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import { createMessageSchema, updateMessageSchema } from "@/types/chat";

export const messageRouter = createTRPCRouter({
	create: protectedProcedure
		.input(createMessageSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			// Check if user is subscribed
			const subscription = await ctx.db.subscription.findFirst({
				where: {
					userId: userId,
					status: "active",
				},
			});
			const isSubscribed = !!subscription;

			// If not subscribed, check daily limit before creating message (skip in development)
			const isDevelopment = process.env.NODE_ENV === "development";
			if (!isSubscribed && !isDevelopment && input.role === "user") {
				const hasReachedLimit = await hasReachedDailyLimit(ctx.db, userId);
				if (hasReachedLimit) {
					throw new TRPCError({
						code: "TOO_MANY_REQUESTS",
						message: "Daily message limit reached. Please upgrade to continue.",
					});
				}
			}

			const { conversationId, ...messageData } = input;

			const conversation = await findConversationByUserAndId(
				ctx.db,
				conversationId,
				userId,
			);
			validateConversationAccess(conversation);

			const messageDataToCreate = createMessageData(
				messageData,
				conversationId,
			);
			return createMessageWithTimestampUpdate(
				ctx.db,
				messageDataToCreate,
				conversationId,
			);
		}),

	listByConversation: protectedProcedure
		.input(z.object({ conversationId: z.number() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			const conversation = await findConversationByUserAndId(
				ctx.db,
				input.conversationId,
				userId,
			);
			validateConversationAccess(conversation);

			return ctx.db.message.findMany({
				where: { conversationId: input.conversationId, deletedAt: null },
				orderBy: { createdAt: "asc" },
			});
		}),

	getById: protectedProcedure
		.input(z.object({ id: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			const message = await findMessageWithConversationAccess(
				ctx.db,
				input.id,
				userId,
			);
			return validateMessageAccess(message);
		}),

	update: protectedProcedure
		.input(updateMessageSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			const { id, ...dataToUpdate } = input;

			const messageResult = await findMessageWithConversationAccess(
				ctx.db,
				id,
				userId,
			);
			const message = validateMessageAccess(messageResult);

			const updateData = updateMessageData(dataToUpdate);
			return updateMessageWithTimestampUpdate(
				ctx.db,
				id,
				updateData,
				message.conversationId,
			);
		}),

	delete: protectedProcedure
		.input(z.object({ id: z.string() }))
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			const messageResult = await findMessageWithConversationAccess(
				ctx.db,
				input.id,
				userId,
			);
			const message = validateMessageAccess(messageResult);

			return softDeleteMessageWithTimestampUpdate(
				ctx.db,
				input.id,
				message.conversationId,
			);
		}),

	batchUpsert: protectedProcedure
		.input(
			z.object({
				conversationId: z.number(),
				messages: z.array(createMessageSchema),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			const { conversationId, messages: messagesData } = input;

			const conversation = await findConversationByUserAndId(
				ctx.db,
				conversationId,
				userId,
			);
			validateConversationAccess(conversation);

			if (!messagesData?.length) {
				return { count: 0 };
			}

			const results = await ctx.db.$transaction(async (tx) => {
				const upsertResults = await Promise.all(
					messagesData.map((messageData) => {
						const createData = {
							id: messageData.id,
							role: messageData.role,
							metadata: messageData.metadata
								? JSON.parse(JSON.stringify(messageData.metadata))
								: null,
							annotations: messageData.annotations
								? JSON.parse(JSON.stringify(messageData.annotations))
								: null,
							parts: JSON.parse(JSON.stringify(messageData.parts)),
							conversation: { connect: { id: conversationId } },
							...(messageData.createdAt && {
								createdAt: new Date(messageData.createdAt),
							}),
						};

						const updateData = {
							metadata: messageData.metadata
								? JSON.parse(JSON.stringify(messageData.metadata))
								: null,
							annotations: messageData.annotations
								? JSON.parse(JSON.stringify(messageData.annotations))
								: null,
							parts: JSON.parse(JSON.stringify(messageData.parts)),
						};

						return tx.message.upsert({
							where: { id: messageData.id },
							create: createData,
							update: updateData,
						});
					}),
				);

				await tx.conversation.update({
					where: { id: conversationId },
					data: { updatedAt: new Date() },
				});

				return upsertResults;
			});

			return { count: results.length };
		}),

	getRemainingDaily: protectedProcedure.query(async ({ ctx }) => {
		const userId = ctx.clerkAuth.userId;

		// Check if user is subscribed
		const subscription = await ctx.db.subscription.findFirst({
			where: {
				userId: userId,
				status: "active",
			},
		});
		const isSubscribed = !!subscription;

		if (isSubscribed) {
			return { unlimited: true, remaining: null };
		}

		const remaining = await getRemainingMessages(ctx.db, userId);
		return { unlimited: false, remaining };
	}),
});
