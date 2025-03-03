import { useState } from "react";
import {
  Message,
  OpenAIResponse,
  AnthropicResponse,
  GroqResponse,
} from "@/services/llms/types";
import { useChatCompletion } from "./use-chat-completion";

/**
 * Extracts the message content from a provider's response
 *
 * @param provider - The provider name
 * @param response - The provider-specific response object
 * @returns The extracted message content
 */
function extractMessageContent(provider: string, response: unknown): string {
  switch (provider.toLowerCase()) {
    case "openai": {
      const openaiResp = response as OpenAIResponse;
      return openaiResp.choices[0]?.message.content || "";
    }

    case "anthropic": {
      const anthropicResp = response as AnthropicResponse;
      return anthropicResp.content.map((item) => item.text).join("") || "";
    }

    case "groq": {
      const groqResp = response as GroqResponse;
      return groqResp.choices[0]?.message.content || "";
    }

    default:
      throw new Error("Unsupported provider.");
  }
}

/**
 * Hook for managing a conversation with the chat API
 *
 * @returns Conversation state and methods to interact with it
 *
 * @example
 * ```tsx
 * const ChatInterface = () => {
 *   const {
 *     messages,
 *     sendMessage,
 *     isLoading,
 *     error,
 *     resetConversation,
 *     lastResponse
 *   } = useConversation();
 *
 *   // You can access the typed response directly
 *   useEffect(() => {
 *     if (lastResponse && lastResponse.provider === 'openai') {
 *       const openaiResponse = lastResponse.response as OpenAIResponse;
 *       console.log('Tokens used:', openaiResponse.usage.total_tokens);
 *     }
 *   }, [lastResponse]);
 *
 *   return (
 *     // UI implementation
 *   );
 * };
 * ```
 */
export const useConversation = (initialMessages: Message[] = []) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<{
    provider: string;
    response: OpenAIResponse | AnthropicResponse | GroqResponse;
  } | null>(null);

  const chatMutation = useChatCompletion();

  /**
   * Sends a user message to the chat API
   *
   * @param content - The message content to send
   */
  const sendMessage = async (content: string) => {
    // Add user message to the conversation
    const userMessage: Message = { role: "user", content };
    setMessages((prev) => [...prev, userMessage]);
    setError(null);

    try {
      // Send all messages to maintain conversation context
      const response = await chatMutation.mutateAsync({
        messages: [...messages, userMessage],
      });

      if (response.error) {
        setError(response.error);
        return;
      }

      // Store the last response for advanced usage
      setLastResponse({
        provider: response.provider,
        response: response.response,
      });

      // Extract content from the response based on provider
      const assistantContent = extractMessageContent(
        response.provider,
        response.response,
      );

      // Add the assistant message to the conversation
      const assistantMessage: Message = {
        role: "assistant",
        content: assistantContent,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  /**
   * Resets the conversation to its initial state
   */
  const resetConversation = () => {
    setMessages(initialMessages);
    setError(null);
    setLastResponse(null);
  };

  return {
    messages,
    sendMessage,
    isLoading: chatMutation.isPending,
    error,
    resetConversation,
    lastResponse,
  };
};
