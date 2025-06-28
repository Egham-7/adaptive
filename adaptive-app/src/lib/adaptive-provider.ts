import {
  type LanguageModelV1,
  type LanguageModelV1CallOptions,
  type LanguageModelV1StreamPart,
} from '@ai-sdk/provider';
import {
  loadApiKey,
  withoutTrailingSlash,
  combineHeaders,
  createEventSourceResponseHandler,
  postJsonToApi,
  createStatusCodeErrorResponseHandler,
  createJsonErrorResponseHandler,
} from '@ai-sdk/provider-utils';
import { z } from 'zod';

// Adaptive provider interface
export interface AdaptiveProvider {
  (
    modelId: AdaptiveModelId,
    settings?: AdaptiveChatSettings
  ): LanguageModelV1;

  chat(
    modelId: AdaptiveModelId,
    settings?: AdaptiveChatSettings
  ): LanguageModelV1;
}

export interface AdaptiveChatSettings {
  /**
   * Base URL for the Adaptive API. If not provided, defaults to localhost.
   */
  baseURL?: string;
  
  /**
   * API key for the Adaptive API.
   */
  apiKey?: string;
  
  /**
   * Additional headers to include in the request.
   */
  headers?: Record<string, string>;
  
  /**
   * Provider to compare costs against for cost_saved calculation.
   */
  compareProvider?: string;
  
  /**
   * Model to compare costs against for cost_saved calculation.
   */
  compareModel?: string;
}

export type AdaptiveModelId = string;

interface AdaptiveProviderOptions {
  baseURL?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  compareProvider?: string;
  compareModel?: string;
}

// Response schemas
const adaptiveErrorDataSchema = z.object({
  error: z.object({
    message: z.string(),
    type: z.string().optional(),
    code: z.string().optional(),
  }),
});

const adaptiveCompletionChunkSchema = z.object({
  id: z.string().optional(),
  object: z.string().optional(),
  created: z.number().optional(),
  model: z.string().optional(),
  provider: z.string().optional(),
  cost_saved: z.number().optional(),
  choices: z.array(
    z.object({
      index: z.number(),
      delta: z.object({
        role: z.string().optional(),
        content: z.string().optional(),
        tool_calls: z.array(z.any()).optional(),
      }),
      finish_reason: z.string().nullish(),
    })
  ).optional(),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
  }).optional(),
});

export function createAdaptiveProvider(
  options: AdaptiveProviderOptions = {},
): AdaptiveProvider {
  const createModel = (
    modelId: AdaptiveModelId,
    settings: AdaptiveChatSettings = {},
  ): LanguageModelV1 => ({
    specificationVersion: 'v1',
    provider: 'adaptive',
    modelId,
    defaultObjectGenerationMode: 'json',
    
    async doGenerate(args: LanguageModelV1CallOptions) {
      const { prompt, mode, ...restOptions } = args;
      
      const baseURL = withoutTrailingSlash(
        settings.baseURL ?? options.baseURL ?? 'http://localhost:8080'
      );
      
      const headers = combineHeaders(
        {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${loadApiKey({
            apiKey: settings.apiKey ?? options.apiKey,
            environmentVariableName: 'ADAPTIVE_API_KEY',
            description: 'Adaptive API key',
          })}`,
        },
        options.headers,
        settings.headers
      );

      // Convert prompt to OpenAI format
      const messages = prompt.map((message) => {
        switch (message.role) {
          case 'system':
            return { role: 'system', content: message.content };
          case 'user':
            return {
              role: 'user',
              content: message.content.map((part) => {
                if (part.type === 'text') {
                  return { type: 'text', text: part.text };
                }
                if (part.type === 'image') {
                  return {
                    type: 'image_url',
                    image_url: {
                      url: part.image instanceof URL ? part.image.toString() : `data:${part.mimeType};base64,${Buffer.from(part.image).toString('base64')}`,
                    },
                  };
                }
                return part;
              }),
            };
          case 'assistant':
            return {
              role: 'assistant',
              content: message.content
                .map((part) => {
                  if (part.type === 'text') {
                    return part.text;
                  }
                  return null;
                })
                .filter(Boolean)
                .join(''),
              tool_calls: message.content
                .filter((part) => part.type === 'tool-call')
                .map((part: any) => ({
                  id: part.toolCallId,
                  type: 'function',
                  function: {
                    name: part.toolName,
                    arguments: JSON.stringify(part.args),
                  },
                })),
            };
          case 'tool':
            return {
              role: 'tool',
              tool_call_id: message.content[0]?.toolCallId,
              content: JSON.stringify(message.content[0]?.result),
            };
          default:
            return message;
        }
      });

      const body = {
        model: modelId,
        messages,
        stream: false,
        ...restOptions,
        // Add cost comparison parameters if provided
        ...(settings.compareProvider && {
          compare_provider: settings.compareProvider ?? options.compareProvider,
        }),
        ...(settings.compareModel && {
          compare_model: settings.compareModel ?? options.compareModel,
        }),
      };

      const { value: response } = await postJsonToApi({
        url: `${baseURL}/v1/chat/completions`,
        headers,
        body,
        failedResponseHandler: createJsonErrorResponseHandler({
          errorSchema: adaptiveErrorDataSchema,
          errorToMessage: (error) => error.error.message,
        }),
        successfulResponseHandler: (response) => response.json(),
        abortSignal: args.abortSignal,
      });

      const choice = response.choices[0];
      
      return {
        text: choice?.message?.content ?? '',
        finishReason: choice?.finish_reason ?? 'stop',
        usage: {
          promptTokens: response.usage?.prompt_tokens ?? 0,
          completionTokens: response.usage?.completion_tokens ?? 0,
        },
        rawCall: {
          rawPrompt: messages,
          rawSettings: body,
        },
        response: {
          id: response.id,
          timestamp: new Date(),
          modelId: response.model ?? modelId,
        },
        // Include adaptive-specific metadata
        ...(response.cost_saved && { 
          providerMetadata: { 
            adaptive: { costSaved: response.cost_saved } 
          }
        }),
      };
    },

    async doStream(args: LanguageModelV1CallOptions) {
      const { prompt, mode, ...restOptions } = args;
      
      const baseURL = withoutTrailingSlash(
        settings.baseURL ?? options.baseURL ?? 'http://localhost:8080'
      );
      
      const headers = combineHeaders(
        {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${loadApiKey({
            apiKey: settings.apiKey ?? options.apiKey,
            environmentVariableName: 'ADAPTIVE_API_KEY',
            description: 'Adaptive API key',
          })}`,
        },
        options.headers,
        settings.headers
      );

      // Convert prompt to OpenAI format (same as above)
      const messages = prompt.map((message) => {
        switch (message.role) {
          case 'system':
            return { role: 'system', content: message.content };
          case 'user':
            return {
              role: 'user',
              content: message.content.map((part) => {
                if (part.type === 'text') {
                  return { type: 'text', text: part.text };
                }
                if (part.type === 'image') {
                  return {
                    type: 'image_url',
                    image_url: {
                      url: part.image instanceof URL ? part.image.toString() : `data:${part.mimeType};base64,${Buffer.from(part.image).toString('base64')}`,
                    },
                  };
                }
                return part;
              }),
            };
          case 'assistant':
            return {
              role: 'assistant',
              content: message.content
                .map((part) => {
                  if (part.type === 'text') {
                    return part.text;
                  }
                  return null;
                })
                .filter(Boolean)
                .join(''),
              tool_calls: message.content
                .filter((part) => part.type === 'tool-call')
                .map((part: any) => ({
                  id: part.toolCallId,
                  type: 'function',
                  function: {
                    name: part.toolName,
                    arguments: JSON.stringify(part.args),
                  },
                })),
            };
          case 'tool':
            return {
              role: 'tool',
              tool_call_id: message.content[0]?.toolCallId,
              content: JSON.stringify(message.content[0]?.result),
            };
          default:
            return message;
        }
      });

      const body = {
        model: modelId,
        messages,
        stream: true,
        ...restOptions,
        // Add cost comparison parameters if provided
        ...(settings.compareProvider && {
          compare_provider: settings.compareProvider ?? options.compareProvider,
        }),
        ...(settings.compareModel && {
          compare_model: settings.compareModel ?? options.compareModel,
        }),
      };

      const { value: responseStream } = await postJsonToApi({
        url: `${baseURL}/v1/chat/completions`,
        headers,
        body,
        failedResponseHandler: createStatusCodeErrorResponseHandler(),
        successfulResponseHandler: createEventSourceResponseHandler(adaptiveCompletionChunkSchema),
        abortSignal: args.abortSignal,
      });

      let finishReason: string | undefined;
      let usage: { promptTokens: number; completionTokens: number } | undefined;
      let costSaved: number | undefined;
      let provider: string | undefined;

      const transformStream = new TransformStream<any, LanguageModelV1StreamPart>({
        transform(chunk, controller) {
          if (chunk.success === false) {
            controller.enqueue({
              type: 'error',
              error: chunk.error,
            });
            return;
          }

          const data = chunk.value;
          
          if (data.choices && data.choices[0]) {
            const choice = data.choices[0];
            
            if (choice.delta?.content) {
              controller.enqueue({
                type: 'text-delta',
                textDelta: choice.delta.content,
              });
            }

            if (choice.finish_reason) {
              finishReason = choice.finish_reason;
            }
          }

          if (data.usage) {
            usage = {
              promptTokens: data.usage.prompt_tokens,
              completionTokens: data.usage.completion_tokens,
            };
          }

          if (data.cost_saved !== undefined) {
            costSaved = data.cost_saved;
          }

          if (data.provider) {
            provider = data.provider;
          }

          // Send response metadata if available
          if (data.id || data.model) {
            controller.enqueue({
              type: 'response-metadata',
              id: data.id,
              modelId: data.model,
              timestamp: new Date(),
            });
          }
        },
        
        flush(controller) {
          if (finishReason && usage) {
            controller.enqueue({
              type: 'finish',
              finishReason: finishReason as any,
              usage,
              // Include adaptive-specific metadata
              ...(costSaved !== undefined && {
                providerMetadata: {
                  adaptive: {
                    costSaved,
                    ...(provider && { provider }),
                  },
                },
              }),
            });
          }
        },
      });

      return {
        stream: responseStream.pipeThrough(transformStream),
        rawCall: {
          rawPrompt: messages,
          rawSettings: body,
        },
      };
    },
  });

  const provider = (
    modelId: AdaptiveModelId,
    settings?: AdaptiveChatSettings,
  ) => {
    if (new.target) {
      throw new Error(
        'The Adaptive provider function cannot be called with the new keyword.',
      );
    }

    return createModel(modelId, settings);
  };

  provider.chat = createModel;

  return provider;
}

/**
 * Default Adaptive provider instance.
 */
export const adaptive = createAdaptiveProvider();