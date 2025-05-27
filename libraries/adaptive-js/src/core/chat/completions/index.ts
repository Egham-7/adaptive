import { $fetch } from "../../../utils/fetch";
import {
  ChatCompletionStreamingResponse,
  ChatAPIResponseSchema,
  ChatAPIResponse,
  ChatCompletionStreamingResponseSchema,
} from "../../../models/chat-completions";
import { Message } from "../../../models/chat-completions";
import { createStreamIterator } from "../../../utils/stream";

export class Completions {
  constructor(
    private apiKey: string,
    private baseUrl: string,
  ) {}

  async create(
    messages: Message[],
    stream = false,
    params: Record<string, unknown> = {},
  ): Promise<
    ChatAPIResponse | AsyncGenerator<ChatCompletionStreamingResponse>
  > {
    const payload = { messages, ...params };

    if (stream) {
      const response = await fetch(
        `${this.baseUrl}/api/chat/completions/stream`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            Accept: "text/event-stream",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
      );

      const reader = response.body?.getReader();

      return createStreamIterator<ChatCompletionStreamingResponse>(reader, {
        schema: ChatCompletionStreamingResponseSchema,
      });
    }

    const data: ChatAPIResponse = await $fetch(
      `${this.baseUrl}/api/chat/completions`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: payload,
        output: ChatAPIResponseSchema,
      },
    );

    return data;
  }
}
