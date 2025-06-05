// src/server/api/routers/message.ts
import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import { createMessageSchema, updateMessageSchema } from "@/lib/chat/schema";

export const messageRouter = createTRPCRouter({
  create: protectedProcedure
    .input(createMessageSchema)
    .mutation(async ({ ctx, input }) => {
      const { conversationId, ...messageData } = input;
      const userId = ctx.clerkAuth.userId;

      // Verify user owns the conversation
      const conversation = await ctx.db.conversation.findUnique({
        where: { id: conversationId, userId: userId, deletedAt: null },
      });

      if (!conversation) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Conversation not found or you do not have access.",
        });
      }

      const messageDataToCreate = {
        ...messageData,
        role: input.role,
        content: input.content,
        conversation: { connect: { id: conversationId } },
        ...(input.createdAt && { createdAt: new Date(input.createdAt) }),
      };

      const newMessage = await ctx.db.message.create({
        data: messageDataToCreate,
      });

      await ctx.db.conversation.update({
        where: { id: conversationId },
        data: { updatedAt: new Date() },
      });

      return newMessage;
    }),

  listByConversation: protectedProcedure
    .input(z.object({ conversationId: z.number() }))
    .query(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      const { conversationId } = input;

      const conversation = await ctx.db.conversation.findFirst({
        where: { id: conversationId, userId: userId, deletedAt: null },
      });

      if (!conversation) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Conversation not found or you do not have access.",
        });
      }

      return ctx.db.message.findMany({
        where: { conversationId: conversationId, deletedAt: null },
        orderBy: { createdAt: "asc" },
      });
    }),

  getById: protectedProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      const message = await ctx.db.message.findFirst({
        where: {
          id: input.id,
          deletedAt: null,
          conversation: {
            userId: userId, // Ensure message belongs to user's conversation
            deletedAt: null,
          },
        },
      });

      if (!message) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Message not found or you do not have access.",
        });
      }
      return message;
    }),

  update: protectedProcedure
    .input(updateMessageSchema)
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      const { id, ...dataToUpdate } = input;

      const existingMessage = await ctx.db.message.findFirst({
        where: {
          id: id,
          deletedAt: null,
          conversation: {
            userId: userId,
            deletedAt: null,
          },
        },
        select: { conversationId: true },
      });

      if (!existingMessage) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Message not found or you do not have access to update it.",
        });
      }

      const updatedMessage = await ctx.db.message.update({
        where: { id },
        data: {
          ...dataToUpdate,
          updatedAt: new Date(),
        },
      });

      await ctx.db.conversation.update({
        where: { id: existingMessage.conversationId },
        data: { updatedAt: new Date() },
      });

      return updatedMessage;
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;

      const messageToDelete = await ctx.db.message.findFirst({
        where: {
          id: input.id,
          deletedAt: null,
          conversation: {
            userId: userId,
            deletedAt: null,
          },
        },
        select: { conversationId: true },
      });

      if (!messageToDelete) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Message not found or you do not have access to delete it.",
        });
      }

      const deletedMessage = await ctx.db.message.update({
        where: { id: input.id },
        data: { deletedAt: new Date() },
      });

      await ctx.db.conversation.update({
        where: { id: messageToDelete.conversationId },
        data: { updatedAt: new Date() },
      });

      return deletedMessage; // Or a success message
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

      if (!messagesData || messagesData.length === 0) {
        return [];
      }

      const conversation = await ctx.db.conversation.findFirst({
        where: { id: conversationId, userId: userId, deletedAt: null },
      });

      if (!conversation) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Conversation not found or you do not have access.",
        });
      }

      const upsertOperations = [];

      for (const msgData of messagesData) {
        const messageToUpsert = {
          role: msgData.role,
          content: msgData.content,
          reasoning: msgData.reasoning,
          data: msgData.data,
          annotations: msgData.annotations,
          toolInvocations: msgData.toolInvocations,
          parts: msgData.parts,
          experimentalAttachments: msgData.experimentalAttachments,
        };
        upsertOperations.push(
          ctx.db.message.upsert({
            where: { id: msgData.id },
            create: {
              id: msgData.id,
              conversationId: conversationId,
              createdAt: msgData.createdAt
                ? new Date(msgData.createdAt)
                : new Date(),
              ...messageToUpsert,
            },
            update: { ...messageToUpsert, updatedAt: new Date() },
          }),
        );
      }

      if (upsertOperations.length === 0) {
        return []; // Should not happen if messagesData is not empty and validated
      }

      const results = await ctx.db.$transaction(upsertOperations);
      await ctx.db.conversation.update({
        where: { id: conversationId },
        data: { updatedAt: new Date() },
      });
      return results;
    }),
});
