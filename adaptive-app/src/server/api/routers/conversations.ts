// src/server/api/routers/conversation.ts
import { z } from "zod";
import { TRPCError } from "@trpc/server";
// Import PrismaClient and Prisma namespace
import { type PrismaClient, Prisma } from "@prisma/client";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import {
  createConversationSchema,
  updateConversationSchema,
  getConversationsOptionsSchema,
} from "@/lib/chat/schema"; // Ensure this path is correct

export const conversationRouter = createTRPCRouter({
  create: protectedProcedure
    .input(createConversationSchema)
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      return ctx.db.conversation.create({
        data: {
          ...input,
          userId: userId, // Associate with the logged-in user
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
        include: {
          messages: input?.includeMessages
            ? {
                where: { deletedAt: null },
                orderBy: { createdAt: "asc" },
                take: 10, // Example: limit messages shown in list view
              }
            : false,
        },
        orderBy: [{ pinned: "desc" }, { updatedAt: "desc" }],
      });
    }),

  update: protectedProcedure
    .input(updateConversationSchema)
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      const { id, ...dataToUpdate } = input;

      // Verify user owns the conversation
      const existingConversation = await ctx.db.conversation.findFirst({
        where: { id: id, userId: userId, deletedAt: null },
      });

      if (!existingConversation) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message:
            "Conversation not found or you do not have access to update it.",
        });
      }

      return ctx.db.conversation.update({
        where: { id: id }, // Unique identifier for update
        data: {
          ...dataToUpdate,
          updatedAt: new Date(),
        },
      });
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.number() }))
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;

      // Verify user owns the conversation
      const conversationToDelete = await ctx.db.conversation.findFirst({
        where: { id: input.id, userId: userId, deletedAt: null },
      });

      if (!conversationToDelete) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message:
            "Conversation not found or you do not have access to delete it.",
        });
      }

      // Soft delete conversation and its messages in a transaction
      // Correctly type 'prisma' using Prisma.TransactionClient
      return ctx.db.$transaction(async (prisma: Prisma.TransactionClient) => {
        const deletedConversation = await prisma.conversation.update({
          where: { id: input.id },
          data: { deletedAt: new Date() },
        });

        await prisma.message.updateMany({
          where: { conversationId: input.id, deletedAt: null },
          data: { deletedAt: new Date() },
        });
        return deletedConversation; // Or a success message
      });
    }),

  pin: protectedProcedure
    .input(z.object({ id: z.number() }))
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      const existingConversation = await ctx.db.conversation.findFirst({
        where: { id: input.id, userId: userId, deletedAt: null },
      });
      if (!existingConversation) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Conversation not found.",
        });
      }
      return ctx.db.conversation.update({
        where: { id: input.id },
        data: { pinned: true, updatedAt: new Date() },
      });
    }),

  unpin: protectedProcedure
    .input(z.object({ id: z.number() }))
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.clerkAuth.userId;
      const existingConversation = await ctx.db.conversation.findFirst({
        where: { id: input.id, userId: userId, deletedAt: null },
      });
      if (!existingConversation) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Conversation not found.",
        });
      }
      return ctx.db.conversation.update({
        where: { id: input.id },
        data: { pinned: false, updatedAt: new Date() },
      });
    }),
});
