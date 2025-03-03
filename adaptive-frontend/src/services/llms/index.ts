import axios from "axios";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  OpenAIResponse,
  AnthropicResponse,
  GroqResponse,
} from "./types";

/**
 * Base URL for API requests
 */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

/**
 * Helper function to type the response based on provider name
 *
 * @param provider - The name of the provider
 * @param response - The response data
 * @returns The properly typed response
 */
function typeProviderResponse(
  provider: string,
  response: unknown,
): OpenAIResponse | AnthropicResponse | GroqResponse {
  switch (provider.toLowerCase()) {
    case "openai":
      return response as OpenAIResponse;
    case "anthropic":
      return response as AnthropicResponse;
    case "groq":
      return response as GroqResponse;
    default:
      throw new Error("Must be a valid provider.");
  }
}

/**
 * Creates a chat completion by sending the user's messages to the API
 *
 * @param request - The chat completion request containing messages
 * @returns Promise with the chat completion response
 */
export const createChatCompletion = async (
  request: ChatCompletionRequest,
): Promise<ChatCompletionResponse> => {
  const response = await axios.post<ChatCompletionResponse>(
    `${API_BASE_URL}/api/chat/completion`,
    request,
  );

  // Ensure the response is properly typed based on the provider
  const typedResponse = {
    ...response.data,
    response: typeProviderResponse(
      response.data.provider,
      response.data.response,
    ),
  };

  return typedResponse;
};

/**
 * React hook for creating chat completions
 *
 * @returns TanStack mutation hook for chat completions
 *
 * @example
 * ```tsx
 * const ChatComponent = () => {
 *   const { mutate, isPending, error, data } = useChatCompletion();
 *
 *   const handleSendMessage = () => {
 *     mutate({
 *       messages: [
 *         { role: 'user', content: 'Hello, how can you help me?' }
 *       ]
 *     });
 *   };
 *
 *   if (data?.provider === 'openai') {
 *     // You now have type-safe access to OpenAI specific fields
 *     const openaiResponse = data.response as OpenAIResponse;
 *     console.log(openaiResponse.choices[0].message.content);
 *   }
 *
 *   return (
 *     <div>
 *       <button onClick={handleSendMessage} disabled={isPending}>
 *         Send Message
 *       </button>
 *       {isPending && <p>Loading...</p>}
 *       {error && <p>Error: {error.message}</p>}
 *       {data && <p>Response from: {data.provider}</p>}
 *     </div>
 *   );
 * }
 * ```
 */
