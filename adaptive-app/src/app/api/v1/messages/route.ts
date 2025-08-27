import Anthropic from "@anthropic-ai/sdk";
import type { NextRequest } from "next/server";
import { api } from "@/trpc/server";
import type { AnthropicMessagesRequest } from "@/types/anthropic-messages";
import { anthropicMessagesRequestSchema } from "@/types/anthropic-messages";
import { env } from "@/env";

export async function POST(req: NextRequest) {
  try {
    const rawBody = await req.json();
    const validationResult = anthropicMessagesRequestSchema.safeParse(rawBody);
    
    if (!validationResult.success) {
      return new Response(
        JSON.stringify({
          type: "error",
          error: {
            type: "validation_error",
            message: "Invalid request body",
            details: validationResult.error.issues,
          },
        }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    
    const body = validationResult.data;

    // Extract API key from headers
    const authHeader = req.headers.get("authorization");
    const bearerToken = authHeader?.startsWith("Bearer ")
      ? authHeader.slice(7).replace(/\s+/g, "") || null
      : null;

    const apiKey =
      bearerToken ||
      req.headers.get("x-api-key") ||
      req.headers.get("api-key") ||
      req.headers.get("x-stainless-api-key");

    if (!apiKey) {
      return new Response(
        JSON.stringify({
          type: "error",
          error: {
            type: "authentication_error",
            message:
              "API key required. Provide it via Authorization: Bearer, X-API-Key, api-key, or X-Stainless-API-Key header",
          },
        }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    // Verify API key
    const verificationResult = await api.api_keys.verify({ apiKey });
    if (!verificationResult.valid) {
      return new Response(
        JSON.stringify({
          type: "error",
          error: {
            type: "authentication_error",
            message: "Invalid API key",
          },
        }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    // Use Anthropic SDK to call our backend
    const anthropic = new Anthropic({
      apiKey: apiKey,
      baseURL: env.ADAPTIVE_API_BASE_URL,
    });

    const startTime = Date.now();

    if (body.stream) {
      // Handle streaming with Anthropic SDK
      const stream = anthropic.messages.stream({
        model: body.model,
        max_tokens: body.max_tokens,
        messages: body.messages,
        ...(body.system && { system: body.system }),
        ...(body.temperature !== undefined && {
          temperature: body.temperature,
        }),
        ...(body.top_p !== undefined && { top_p: body.top_p }),
        ...(body.top_k !== undefined && { top_k: body.top_k }),
        ...(body.stop_sequences && { stop_sequences: body.stop_sequences }),
        ...(body.metadata && { metadata: body.metadata }),
        ...(body.tools && { tools: body.tools }),
        ...(body.tool_choice && { tool_choice: body.tool_choice }),
        // Custom adaptive extensions (cast to bypass SDK type checking)
        ...(body.provider_configs && {
          provider_configs: body.provider_configs,
        }),
        ...(body.model_router && {
          model_router: body.model_router,
        }),
        ...(body.semantic_cache && { 
          prompt_response_cache: { 
            enabled: body.semantic_cache.enabled,
            semantic_threshold: body.semantic_cache.semantic_threshold 
          } 
        }),
        ...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
        ...(body.fallback && { fallback: body.fallback }),
      } as Anthropic.MessageStreamParams);

      // Create a ReadableStream that forwards the Anthropic stream
      const readableStream = new ReadableStream({
        async start(controller) {
          let finalMessage: Anthropic.Message | null = null;
          
          try {
            for await (const chunk of stream) {
              const data = `data: ${JSON.stringify(chunk)}\n\n`;
              controller.enqueue(new TextEncoder().encode(data));
            }

            // Try to get final message for usage recording
            try {
              finalMessage = await stream.finalMessage();
            } catch (finalError) {
              console.warn("Could not get final message for usage recording:", finalError);
            }

            // Send final data event to close the stream
            controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
            controller.close();

            // Record usage if available
            if (finalMessage?.usage) {
              setImmediate(async () => {
                try {
                  await api.usage.recordApiUsage({
                    apiKey,
                    provider: "anthropic", // Default since we're using Anthropic format
                    model: finalMessage!.model,
                    usage: {
                      promptTokens: finalMessage!.usage!.input_tokens,
                      completionTokens: finalMessage!.usage!.output_tokens,
                      totalTokens:
                        finalMessage!.usage!.input_tokens +
                        finalMessage!.usage!.output_tokens,
                    },
                    duration: Date.now() - startTime,
                    timestamp: new Date(),
                  });
                } catch (error) {
                  console.error("Failed to record usage:", error);
                }
              });
            }
          } catch (error) {
            console.error("Stream error:", error);
            // Send error event before closing
            const errorData = `data: ${JSON.stringify({ type: "error", error: { message: "Stream failed" } })}\n\n`;
            controller.enqueue(new TextEncoder().encode(errorData));
            controller.close();
          }
        },
      });

      return new Response(readableStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }
    // Handle non-streaming with Anthropic SDK
    const message = (await anthropic.messages.create({
      model: body.model,
      max_tokens: body.max_tokens,
      messages: body.messages,
      ...(body.system && { system: body.system }),
      ...(body.temperature !== undefined && {
        temperature: body.temperature,
      }),
      ...(body.top_p !== undefined && { top_p: body.top_p }),
      ...(body.top_k !== undefined && { top_k: body.top_k }),
      ...(body.stop_sequences && { stop_sequences: body.stop_sequences }),
      ...(body.metadata && { metadata: body.metadata }),
      ...(body.tools && { tools: body.tools }),
      ...(body.tool_choice && { tool_choice: body.tool_choice }),
      // Custom adaptive extensions (cast to bypass SDK type checking)
      ...(body.provider_configs && {
        provider_configs: body.provider_configs,
      }),
      ...(body.model_router && {
        model_router: body.model_router,
      }),
      ...(body.semantic_cache && { semantic_cache: body.semantic_cache }),
      ...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
      ...(body.fallback && { fallback: body.fallback }),
    } as Anthropic.MessageCreateParams)) as Anthropic.Message;

    // Record usage
    if (message.usage) {
      setImmediate(async () => {
        try {
          await api.usage.recordApiUsage({
            apiKey,
            provider: "anthropic", // Default since we're using Anthropic format
            model: message.model,
            usage: {
              promptTokens: message.usage.input_tokens,
              completionTokens: message.usage.output_tokens,
              totalTokens:
                message.usage.input_tokens + message.usage.output_tokens,
            },
            duration: Date.now() - startTime,
            timestamp: new Date(),
          });
        } catch (error) {
          console.error("Failed to record usage:", error);
        }
      });
    }

		return Response.json(message);
	} catch (error) {
		console.error("Anthropic Messages API error:", error);
		return new Response(
			JSON.stringify({
				type: "error",
				error: {
					type: "api_error",
					message: "Internal server error",
				},
			}),
			{
				status: 500,
				headers: { "Content-Type": "application/json" },
			},
		);
	}
}
