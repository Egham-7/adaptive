import axios from "axios";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  OpenAIResponse,
  GroqResponse,
  DeepSeekResponse,
  StreamingResponse,
  isStreamingResponseComplete,
} from "./types";
import { API_BASE_URL } from "../common";

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

export const createChatCompletion = async (
  request: ChatCompletionRequest,
): Promise<ChatCompletionResponse> => {
  const { data } = await axios.post<ChatCompletionResponse>(
    `${API_BASE_URL}/api/chat/completions`,
    request,
  );
  return {
    ...data,
    response: typeProviderResponse(data.provider, data.response),
  };
};

export const createStreamingChatCompletion = (
  request: ChatCompletionRequest,
  onChunk: (chunk: StreamingResponse) => void,
  onComplete?: () => void,
  onError?: (error: Error) => void,
): (() => void) => {
  const controller = new AbortController();

  (async () => {
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("Failed to get response reader");

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          onComplete?.();
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        chunk
          .split("\n\n")
          .filter((line) => line.trim().startsWith("data: "))
          .forEach((line) => {
            try {
              const dataContent = JSON.parse(line.slice(6));
              if (isStreamingResponseComplete(dataContent)) {
                onComplete?.();
                return;
              }
              onChunk(dataContent);
            } catch (err) {
              console.warn("SSE chunk parse/callback error:", err);
            }
          });
      }
    } catch (err: any) {
      if (err.name !== "AbortError" && onError) {
        onError(err instanceof Error ? err : new Error(String(err)));
      }
    }
  })();

  return () => controller.abort();
};
