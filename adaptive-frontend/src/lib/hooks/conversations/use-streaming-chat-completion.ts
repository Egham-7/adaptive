import { useMutation, UseMutationResult } from "@tanstack/react-query";
import {
  ChatCompletionRequest,
  StreamingResponse,
} from "@/services/llms/types";
import { createStreamingChatCompletion } from "@/services/llms";

/**
 * Parameters for streaming chat completion
 */
export type StreamingChatCompletionParams = {
  /** The request to send to the LLM */
  request: ChatCompletionRequest;
  /** Callback for each chunk of the streaming response */
  onChunk: (chunk: StreamingResponse) => void;
  /** Callback when streaming is complete */
  onComplete?: () => void;
  /** Callback when an error occurs during streaming */
  onError?: (error: Error) => void;
};

/**
 * Custom hook for streaming chat completions
 *
 * @returns A mutation object that returns an abort function when successful
 */
export const useStreamingChatCompletion = (): UseMutationResult<
  () => void, // Return type is the abort function
  Error,
  StreamingChatCompletionParams
> => {
  return useMutation({
    mutationFn: async ({
      request,
      onChunk,
      onComplete,
      onError,
    }: StreamingChatCompletionParams) => {
      try {
        // Start the streaming process and get the abort function
        const abortFn = createStreamingChatCompletion(
          request,
          onChunk,
          onComplete,
          (error) => {
            // Handle streaming errors
            console.error("Streaming error:", error);
            if (onError) onError(error);
          },
        );

        // Return the abort function
        return abortFn;
      } catch (error) {
        // Handle setup errors
        const formattedError =
          error instanceof Error ? error : new Error(String(error));

        console.error("Chat completion setup error:", formattedError);

        if (onError) {
          onError(formattedError);
        }

        throw formattedError;
      }
    },

    // This handles errors from the mutation itself (not from streaming)
    onError: (error) => {
      console.error("Chat completion mutation error:", error);
    },
  });
};
