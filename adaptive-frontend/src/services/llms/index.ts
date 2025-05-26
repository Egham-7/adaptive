import axios from "axios";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  OpenAIResponse,
  GroqResponse,
  DeepSeekResponse,
  StreamingResponse,
  AnthropicResponse,
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
): OpenAIResponse | GroqResponse | DeepSeekResponse | AnthropicResponse {
  switch (provider.toLowerCase()) {
    case "openai":
      return response as OpenAIResponse;
    case "groq":
      return response as GroqResponse;
    case "deepseek":
      return response as DeepSeekResponse;
    case "anthropic":
      return response as AnthropicResponse;
    default:
      throw new Error("Must be a valid provider.");
  }
}

/**
 * Wraps an API POST request for chat completions.
 *
 * @param request - The chat completion request containing messages
 * @returns Promise with the chat completion response
 */
export const createChatCompletion = async (
  request: ChatCompletionRequest,
): Promise<ChatCompletionResponse> => {
  const response = await axios.post<ChatCompletionResponse>(
    `${API_BASE_URL}/api/chat/completions`,
    request,
  );

  return {
    ...response.data,
    response: typeProviderResponse(
      response.data.provider,
      response.data.response,
    ),
  };
};

/**
 * Splits raw SSE data into message lines.
 */
const splitSSELines = (chunk: string): string[] =>
  chunk.split("\n\n").filter((line) => line.trim().startsWith("data: "));

/**
 * Parses an SSE line into JSON, or returns undefined if parsing fails.
 */
const parseSSEData = (line: string): unknown | undefined => {
  try {
    return JSON.parse(line.substring(6));
  } catch {
    return undefined;
  }
};

/**
 * Handles one SSE chunk: splits, parses, filters, and returns streaming responses.
 */
const extractStreamingResponses = (chunk: string): StreamingResponse[] =>
  splitSSELines(chunk)
    .map(parseSSEData)
    .filter(
      (data): data is StreamingResponse => !!data && typeof data === "object",
    );

/**
 * Checks if the streaming response signals completion.
 */
const isStreamingResponseComplete = (data: unknown): boolean =>
  data === "[DONE]";

/**
 * Processes an SSE response stream as an async generator.
 */
async function* streamSSE(
  response: Response,
  decoder = new TextDecoder(),
): AsyncGenerator<StreamingResponse, void, unknown> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Failed to get reader from response");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const data of extractStreamingResponses(chunk)) {
      yield data;
    }
  }
}

/**
 * Creates a streaming chat completion and processes the events functionally.
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
  const controller = new AbortController();

  const processStream = async () => {
    try {
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

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      for await (const dataContent of streamSSE(response)) {
        if (isStreamingResponseComplete(dataContent)) {
          onComplete?.();
          return;
        }
        try {
          onChunk(dataContent);
        } catch (err) {
          console.warn("Error processing SSE chunk:", err);
        }
      }

      onComplete?.();
    } catch (error: any) {
      if (error.name !== "AbortError" && onError) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    }
  };

  // Start processing stream
  processStream();

  return () => controller.abort();
};
