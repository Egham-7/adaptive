import { createMessageSchema, updateMessageSchema } from "@/lib/chat/schema";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import type { PrismaClient } from "@prisma/client";
import { TRPCError } from "@trpc/server";
import type { Conversation, Message } from "prisma/generated";
import { z } from "zod";

type CreateMessageInput = z.infer<typeof createMessageSchema>;
type UpdateMessageInput = z.infer<typeof updateMessageSchema>;

// Validation functions
const validateConversationAccess = (conversation: Conversation | null) => {
	if (!conversation) {
		throw new TRPCError({
			code: "NOT_FOUND",
			message: "Conversation not found or you do not have access.",
		});
	}
	return conversation;
};

const validateMessageAccess = (message: Message | null) => {
	if (!message) {
		throw new TRPCError({
			code: "NOT_FOUND",
			message: "Message not found or you do not have access.",
		});
	}
	return message;
};

const validateUserId = (userId: string | null | undefined) => {
	if (!userId) {
		throw new TRPCError({
			code: "UNAUTHORIZED",
			message: "User authentication required.",
		});
	}
	return userId;
};

// Data transformation functions
const createMessageData = (
	input: Omit<CreateMessageInput, "conversationId">,
	conversationId: number,
) => ({
	...input,
	conversation: { connect: { id: conversationId } },
	...(input.createdAt && { createdAt: new Date(input.createdAt) }),
});

const updateMessageData = (input: Omit<UpdateMessageInput, "id">) => ({
	...input,
	updatedAt: new Date(),
});

// Database operations
const findConversationByUserAndId = (
	db: PrismaClient,
	conversationId: number,
	userId: string,
) =>
	db.conversation.findUnique({
		where: { id: conversationId, userId, deletedAt: null },
	});

const findMessageWithConversationAccess = (
	db: PrismaClient,
	messageId: string,
	userId: string,
) =>
	db.message.findFirst({
		where: {
			id: messageId,
			deletedAt: null,
			conversation: { userId, deletedAt: null },
		},
	});

const getMessagesByConversation = (db: PrismaClient, conversationId: number) =>
	db.message.findMany({
		where: { conversationId, deletedAt: null },
		orderBy: { createdAt: "asc" },
	});

// Composed operations
const createMessageWithTimestampUpdate = async (
	db: PrismaClient,
	messageData: ReturnType<typeof createMessageData>,
	conversationId: number,
) => {
	return db.$transaction(async (tx: PrismaClient) => {
		const newMessage = await tx.message.create({ data: messageData });
		await tx.conversation.update({
			where: { id: conversationId },
			data: { updatedAt: new Date() },
		});
		return newMessage;
	});
};

const updateMessageWithTimestampUpdate = async (
	db: PrismaClient,
	messageId: string,
	updateData: ReturnType<typeof updateMessageData>,
	conversationId: number,
) => {
	return db.$transaction(async (tx: PrismaClient) => {
		const updatedMessage = await tx.message.update({
			where: { id: messageId },
			data: updateData,
		});
		await tx.conversation.update({
			where: { id: conversationId },
			data: { updatedAt: new Date() },
		});
		return updatedMessage;
	});
};

const deleteMessageWithTimestampUpdate = async (
	db: PrismaClient,
	messageId: string,
	conversationId: number,
) => {
	return db.$transaction(async (tx: PrismaClient) => {
		const deletedMessage = await tx.message.update({
			where: { id: messageId },
			data: { deletedAt: new Date() },
		});
		await tx.conversation.update({
			where: { id: conversationId },
			data: { updatedAt: new Date() },
		});
		return deletedMessage;
	});
};

export const messageRouter = createTRPCRouter({
	create: protectedProcedure
		.input(createMessageSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = validateUserId(ctx.clerkAuth.userId);
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
			const userId = validateUserId(ctx.clerkAuth.userId);

			const conversation = await findConversationByUserAndId(
				ctx.db,
				input.conversationId,
				userId,
			);
			validateConversationAccess(conversation);

			return getMessagesByConversation(ctx.db, input.conversationId);
		}),

	getById: protectedProcedure
		.input(z.object({ id: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = validateUserId(ctx.clerkAuth.userId);

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
			const userId = validateUserId(ctx.clerkAuth.userId);
			const { id, ...dataToUpdate } = input;

			const message = await findMessageWithConversationAccess(
				ctx.db,
				id,
				userId,
			);
			validateMessageAccess(message);

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
			const userId = validateUserId(ctx.clerkAuth.userId);

			const message = await findMessageWithConversationAccess(
				ctx.db,
				input.id,
				userId,
			);
			validateMessageAccess(message);

			return deleteMessageWithTimestampUpdate(
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
			const userId = validateUserId(ctx.clerkAuth.userId);
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
						const { conversationId: _, ...dataWithoutConversationId } =
							messageData;
						return tx.message.upsert({
							where: { id: messageData.id },
							create: {
								...dataWithoutConversationId,
								conversation: { connect: { id: conversationId } },
							},
							update: {
								content: messageData.content,
								reasoning: messageData.reasoning,
								annotations: messageData.annotations,
								parts: messageData.parts,
								experimentalAttachments: messageData.experimentalAttachments,
							},
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
});
