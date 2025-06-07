import { createMessageSchema, updateMessageSchema } from "@/lib/chat/schema";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import type { PrismaClient } from "@prisma/client";
import { TRPCError } from "@trpc/server";
import type { Conversation, Message } from "prisma/generated";
import { z } from "zod";

type CreateMessageInput = z.infer<typeof createMessageSchema>;
type UpdateMessageInput = z.infer<typeof updateMessageSchema>;

// Pure functions for validation and transformation - using function declarations
function validateConversationAccess(
	conversation: Conversation | null,
): asserts conversation is Conversation {
	if (!conversation) {
		throw new TRPCError({
			code: "NOT_FOUND",
			message: "Conversation not found or you do not have access.",
		});
	}
}

function validateMessageAccess(
	message: Message | null,
): asserts message is Message {
	if (!message) {
		throw new TRPCError({
			code: "NOT_FOUND",
			message: "Message not found or you do not have access.",
		});
	}
}

const createMessageData = (
	input: Omit<CreateMessageInput, "conversationId">,
	conversationId: number,
) => ({
	...input,
	role: input.role,
	content: input.content,
	conversation: { connect: { id: conversationId } },
	...(input.createdAt && { createdAt: new Date(input.createdAt) }),
});

const updateMessageData = (input: Omit<UpdateMessageInput, "id">) => ({
	...input,
	updatedAt: new Date(),
});

// Database operations as pure functions (return promises)
const findConversationByUserAndId = (
	db: PrismaClient,
	conversationId: number,
	userId: string,
): Promise<Conversation | null> =>
	db.conversation.findUnique({
		where: { id: conversationId, userId, deletedAt: null },
	});

const findMessageWithConversationAccess = (
	db: PrismaClient,
	messageId: string,
	userId: string,
): Promise<Message | null> =>
	db.message.findFirst({
		where: {
			id: messageId,
			deletedAt: null,
			conversation: {
				userId,
				deletedAt: null,
			},
		},
	});

const createMessage = (
	db: PrismaClient,
	data: ReturnType<typeof createMessageData>,
): Promise<Message> => db.message.create({ data });

const updateConversationTimestamp = (
	db: PrismaClient,
	conversationId: number,
): Promise<Conversation> =>
	db.conversation.update({
		where: { id: conversationId },
		data: { updatedAt: new Date() },
	});

const getMessagesByConversation = (
	db: PrismaClient,
	conversationId: number,
): Promise<Message[]> =>
	db.message.findMany({
		where: { conversationId, deletedAt: null },
		orderBy: { createdAt: "asc" },
	});

const updateMessage = (
	db: PrismaClient,
	id: string,
	data: ReturnType<typeof updateMessageData>,
): Promise<Message> =>
	db.message.update({
		where: { id },
		data,
	});

const softDeleteMessage = (db: PrismaClient, id: string): Promise<Message> =>
	db.message.update({
		where: { id },
		data: { deletedAt: new Date() },
	});

const upsertMessage = (
	db: PrismaClient,
	messageData: CreateMessageInput,
	conversationId: number,
): Promise<Message> => {
	const { conversationId: msgConversationId, ...dataWithoutConversationId } =
		messageData;

	return db.message.upsert({
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
};

// Context type (you may need to adjust this based on your actual context type)
interface TRPCContext {
	db: PrismaClient;
	clerkAuth: {
		userId: string;
	};
}

// Higher-order function for conversation access validation
const withConversationAccess = <T extends { conversationId: number }>(
	handler: (params: {
		ctx: TRPCContext;
		input: T;
		conversation: Conversation;
	}) => Promise<unknown>,
) => {
	return async ({ ctx, input }: { ctx: TRPCContext; input: T }) => {
		const conversation = await findConversationByUserAndId(
			ctx.db,
			input.conversationId,
			ctx.clerkAuth.userId,
		);
		validateConversationAccess(conversation);
		return handler({ ctx, input, conversation });
	};
};

// Higher-order function for message access validation
const withMessageAccess = <T extends { id: string }>(
	handler: (params: {
		ctx: TRPCContext;
		input: T;
		message: Message;
	}) => Promise<unknown>,
) => {
	return async ({ ctx, input }: { ctx: TRPCContext; input: T }) => {
		const message = await findMessageWithConversationAccess(
			ctx.db,
			input.id,
			ctx.clerkAuth.userId,
		);
		validateMessageAccess(message);
		return handler({ ctx, input, message });
	};
};

// Compose database operations
const createMessageWithTimestampUpdate = async (
	db: PrismaClient,
	messageData: ReturnType<typeof createMessageData>,
	conversationId: number,
): Promise<Message> => {
	const newMessage = await createMessage(db, messageData);
	await updateConversationTimestamp(db, conversationId);
	return newMessage;
};

const updateMessageWithTimestampUpdate = async (
	db: PrismaClient,
	messageId: string,
	updateData: ReturnType<typeof updateMessageData>,
	conversationId: number,
): Promise<Message> => {
	const updatedMessage = await updateMessage(db, messageId, updateData);
	await updateConversationTimestamp(db, conversationId);
	return updatedMessage;
};

const deleteMessageWithTimestampUpdate = async (
	db: PrismaClient,
	messageId: string,
	conversationId: number,
): Promise<Message> => {
	const deletedMessage = await softDeleteMessage(db, messageId);
	await updateConversationTimestamp(db, conversationId);
	return deletedMessage;
};

export const messageRouter = createTRPCRouter({
	create: protectedProcedure.input(createMessageSchema).mutation(
		withConversationAccess(async ({ ctx, input }) => {
			const { conversationId, ...messageData } = input;
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
	),

	listByConversation: protectedProcedure
		.input(z.object({ conversationId: z.number() }))
		.query(
			withConversationAccess(async ({ ctx, input }) => {
				return getMessagesByConversation(ctx.db, input.conversationId);
			}),
		),

	getById: protectedProcedure.input(z.object({ id: z.string() })).query(
		withMessageAccess(async ({ message }) => {
			return message;
		}),
	),

	update: protectedProcedure.input(updateMessageSchema).mutation(
		withMessageAccess(async ({ ctx, input, message }) => {
			const { id, ...dataToUpdate } = input;
			const updateData = updateMessageData(dataToUpdate);

			return updateMessageWithTimestampUpdate(
				ctx.db,
				id,
				updateData,
				message.conversationId,
			);
		}),
	),

	delete: protectedProcedure.input(z.object({ id: z.string() })).mutation(
		withMessageAccess(async ({ ctx, input, message }) => {
			return deleteMessageWithTimestampUpdate(
				ctx.db,
				input.id,
				message.conversationId,
			);
		}),
	),

	batchUpsert: protectedProcedure
		.input(
			z.object({
				conversationId: z.number(),
				messages: z.array(createMessageSchema),
			}),
		)
		.mutation(
			withConversationAccess(async ({ ctx, input }) => {
				const { conversationId, messages: messagesData } = input;

				if (!messagesData?.length) {
					return { count: 0 };
				}

				const results = await Promise.all(
					messagesData.map((messageData) =>
						upsertMessage(ctx.db, messageData, conversationId),
					),
				);

				await updateConversationTimestamp(ctx.db, conversationId);
				return { count: results.length };
			}),
		),
});
