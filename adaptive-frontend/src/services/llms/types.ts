/**
 * Represents a chat message role
 */
export type MessageRole = "user" | "assistant" | "system" | "tool";

/**
 * Represents a single chat message
 */
export interface Message {
  role: MessageRole;
  content: string;
  tool_call_id?: string;
  function_call?: {
    name: string;
    arguments: string;
  };
  tool_calls?: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
}

/**
 * Request payload for chat completion
 */
export interface ChatCompletionRequest {
  messages: Message[];
}

/**
 * OpenAI style response object for chat completions
 */
export interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      function_call?: {
        name: string;
        arguments: string;
      };
      tool_calls?: Array<{
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    finish_reason:
      | "stop"
      | "length"
      | "function_call"
      | "content_filter"
      | "tool_calls";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Anthropic style response object for chat completions
 */
export interface AnthropicResponse {
  id: string;
  type: string;
  role: string;
  content: Array<{
    type: string;
    text: string;
  }>;
  model: string;
  stop_reason: string;
  stop_sequence?: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

/**
 * Groq style response object for chat completions
 */
export interface GroqResponse {
  id: string;
  model: string;
  created: number;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * DeepSeek style response object for chat completions
 */
export interface DeepSeekResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      tool_call_id?: string;
    };
    logprobs?: {
      token_logprobs: number[];
      tokens: string[];
      top_logprobs: Array<Record<string, number>>;
    };
    finish_reason: "stop" | "length" | "tool_calls" | "content_filter";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Union type for various provider responses
 */
export type ProviderResponse =
  | OpenAIResponse
  | AnthropicResponse
  | GroqResponse
  | DeepSeekResponse;

/**
 * Response from chat completion API
 */
export interface ChatCompletionResponse {
  provider: string;
  response: ProviderResponse;
  error?: string;
}

/**
 * OpenAI streaming response as returned by the Go code
 */
export interface OpenAIStreamingResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      content?: string;
      role?: string;
      function_call?: {
        name?: string;
        arguments?: string;
      };
      tool_calls?: Array<{
        id?: string;
        type?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason: string | null;
  }>;
  provider: string;
}

/**
 * Groq streaming response as returned by the Go code
 */
export interface GroqStreamingResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      content?: string;
      role?: string;
    };
    finish_reason: string | null;
  }>;
  provider: string;
}

/**
 * DeepSeek streaming response as returned by the Go code
 */
export interface DeepSeekStreamingResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta?: {
      content?: string;
      role?: string;
    };
    finish_reason: string | null;
  }>;
  provider: string;
}

/**
 * Error response format
 */
export interface ErrorResponse {
  error: string;
}

/**
 * Union type for all supported streaming response types
 */
export type StreamingResponse =
  | OpenAIStreamingResponse
  | GroqStreamingResponse
  | DeepSeekStreamingResponse
  | ErrorResponse;

/**
 * Type guard for error responses
 */
export function isErrorResponse(
  response: StreamingResponse,
): response is ErrorResponse {
  return "error" in response;
}

/**
 * Type guard functions to narrow down the specific response type
 */
export function isOpenAIStreamingResponse(
  response: StreamingResponse,
): response is OpenAIStreamingResponse {
  if (isErrorResponse(response)) return false;
  return response.provider === "openai";
}

export function isGroqStreamingResponse(
  response: StreamingResponse,
): response is GroqStreamingResponse {
  if (isErrorResponse(response)) return false;
  return response.provider === "groq";
}

export function isDeepSeekStreamingResponse(
  response: StreamingResponse,
): response is DeepSeekStreamingResponse {
  if (isErrorResponse(response)) return false;
  return response.provider === "deepseek";
}

/**
 * Helper to extract content from any streaming response type
 */
export function extractContentFromStreamingResponse(
  chunk: StreamingResponse,
): string {
  if (isErrorResponse(chunk)) {
    return "";
  }

  if (isOpenAIStreamingResponse(chunk) || isGroqStreamingResponse(chunk)) {
    return chunk.choices[0]?.delta?.content || "";
  } else if (isDeepSeekStreamingResponse(chunk)) {
    return chunk.choices[0]?.delta?.content || "";
  }

  return "";
}
