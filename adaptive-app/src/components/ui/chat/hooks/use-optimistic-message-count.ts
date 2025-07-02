import { useEffect, useRef, useState } from "react";
import type { UIMessage } from "@ai-sdk/react";
import { getUserMessageCount } from "../utils/message-utils";

/**
 * Hook for optimistic message count tracking
 * Provides immediate UI feedback when users send messages
 */
export function useOptimisticMessageCount(
  messages: UIMessage[],
  remainingMessages?: number,
) {
  const [optimisticRemainingMessages, setOptimisticRemainingMessages] =
    useState(remainingMessages);
  const messagesRef = useRef(messages);

  useEffect(() => {
    setOptimisticRemainingMessages(remainingMessages);
  }, [remainingMessages]);

  useEffect(() => {
    if (
      messagesRef.current.length < messages.length &&
      remainingMessages !== undefined
    ) {
      const newUserMessages = getUserMessageCount(messages);
      const oldUserMessages = getUserMessageCount(messagesRef.current);

      if (newUserMessages > oldUserMessages) {
        setOptimisticRemainingMessages((prev) => Math.max(0, (prev || 0) - 1));
      }
    }
    messagesRef.current = messages;
  }, [messages, remainingMessages]);

  return optimisticRemainingMessages;
}