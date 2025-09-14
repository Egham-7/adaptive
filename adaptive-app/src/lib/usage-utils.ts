import type {
	ChatCompletion,
	ChatCompletionChunk,
	ChatCompletionRequest,
} from "@/types/chat-completion";

/**
 * Checks if the user requested usage data in the stream_options
 */
export const userRequestedUsage = (body: ChatCompletionRequest): boolean => {
	return body.stream_options?.include_usage === true;
};

/**
 * Adds usage tracking to the request body for internal processing
 */
export const withUsageTracking = (
	requestBody: ChatCompletionRequest,
): ChatCompletionRequest => ({
	...requestBody,
	stream_options: {
		...requestBody.stream_options,
		include_usage: true,
	},
});

/**
 * Filters usage information from chat completion based on whether it should be included
 */
export const filterUsageFromCompletion = (
	completion: ChatCompletion,
	includeUsage: boolean,
): ChatCompletion =>
	includeUsage ? completion : { ...completion, usage: undefined };

/**
 * Filters usage information from chat completion chunk based on whether it should be included
 */
export const filterUsageFromChunk = (
	chunk: ChatCompletionChunk,
	includeUsage: boolean,
): ChatCompletionChunk =>
	includeUsage ? chunk : { ...chunk, usage: undefined };
