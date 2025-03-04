import { useMutation } from "@tanstack/react-query";
import { createChatCompletion } from "@/services/llms";

export const useChatCompletion = () => {
  return useMutation({
    mutationFn: createChatCompletion,
    onError: (error) => {
      console.error("Chat completion error:", error);
    },
  });
};
