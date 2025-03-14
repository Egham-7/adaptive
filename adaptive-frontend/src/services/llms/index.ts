import axios from "axios";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  OpenAIResponse,
  GroqResponse,
  DeepSeekResponse,
  StreamingResponse,
} from "./types";
import { API_BASE_URL } from "../common";

/**
 * Helper function to type the response based on provider name
 *
 * @param provider - The name of the provider
 * @param response - The response data
 * @returns The properly typed response
 */
export function typeProviderResponse(
  provider: string,
  response: unknown,
): OpenAIResponse | GroqResponse | DeepSeekResponse {
  switch (provider.toLowerCase()) {
    case "openai":
      return response as OpenAIResponse;
    case "groq":
      return response as GroqResponse;
    case "deepseek":
      return response as DeepSeekResponse;
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
  // Fixed URL to match the API routes
  const response = await axios.post<ChatCompletionResponse>(
    `${API_BASE_URL}/api/chat/completions`,
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
 * Creates a streaming chat completion and processes the events
 *
 * @param request - The chat completion request containing messages
 * @param onChunk - Callback function that receives each chunk of the stream
 * @param onComplete - Optional callback function called when the stream completes
 * @param onError - Optional callback function called when an error occurs
 * @returns A function that can be called to abort the stream
 */
export const createStreamingChatCompletion = (
  request: ChatCompletionRequest,
  onChunk: (chunk: StreamingResponse) => void,
  onComplete?: () => void,
  onError?: (error: Error) => void,
): (() => void) => {
  // Create AbortController to allow cancellation
  const controller = new AbortController();

  // Start the stream processing
  (async () => {
    try {
      // Make request with appropriate headers for SSE
      const response = await fetch(
        `${API_BASE_URL}/api/chat/completions/stream`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(request),
          signal: controller.signal,
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Create reader from the response body stream
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get reader from response");
      }

      // Create a TextDecoder to convert Uint8Array to string
      const decoder = new TextDecoder();

      // Process the stream
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          if (onComplete) onComplete();
          break;
        }

        // Decode the chunk and split into lines
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n\n");

        // Process each SSE message
        for (const line of lines) {
          if (!line.trim() || !line.startsWith("data: ")) continue;

          const dataContent = line.substring(6); // Remove 'data: ' prefix

          // Check for the special [DONE] message
          if (dataContent.trim() === "[DONE]") {
            if (onComplete) onComplete();
            continue;
          }

          try {
            // Parse the JSON content and pass to callback
            const jsonData = JSON.parse(dataContent) as StreamingResponse;
            onChunk(jsonData);
          } catch (e) {
            console.warn("Error parsing SSE message:", e);
          }
        }
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.name !== "AbortError" && onError) {
          onError(error instanceof Error ? error : new Error(String(error)));
        }
      }
    }
  })();

  // Return abort function
  return () => controller.abort();
};
