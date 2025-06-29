import { useEffect, useRef, useState } from "react";
import type { UIMessage } from "@ai-sdk/react";

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
      const newUserMessages = messages.filter((m) => m.role === "user").length;
      const oldUserMessages = messagesRef.current.filter(
        (m) => m.role === "user",
      ).length;

      if (newUserMessages > oldUserMessages) {
        setOptimisticRemainingMessages((prev) => Math.max(0, (prev || 0) - 1));
      }
    }
    messagesRef.current = messages;
  }, [messages, remainingMessages]);

  return optimisticRemainingMessages;
}