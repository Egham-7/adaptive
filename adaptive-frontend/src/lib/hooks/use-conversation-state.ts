import { useState } from "react";

interface UseConversationStateOptions {
  conversationId: number;
  initialTitle?: string;
}

export const useConversationState = (options: UseConversationStateOptions) => {
  const [title, setTitle] = useState(
    options.initialTitle || "New Conversation",
  );

  return {
    title,
    setTitle,
  };
};
