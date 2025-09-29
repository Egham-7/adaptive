/**
 * Gemini API utilities for handling URL formats and model extraction
 */

/**
 * Extracts the model name from Gemini colon syntax URL parameters
 * Handles both formats:
 * - "model:generateContent" -> "model"
 * - "model:streamGenerateContent" -> "model"
 * - "model" -> "model" (passthrough)
 *
 * @param modelParam The model parameter from the URL
 * @returns The extracted model name
 * @throws Response with 400 status if parsing fails
 */
export function extractModelFromGeminiParam(modelParam: string): string {
	if (!modelParam.includes(":")) {
		return modelParam;
	}

	const parts = modelParam.split(":");
	const model = parts[0];

	if (!model || model.trim() === "") {
		throw new Response(
			JSON.stringify({
				error: {
					code: 400,
					message:
						"Invalid model format in URL. Expected 'model:method' or 'model'",
					status: "INVALID_ARGUMENT",
				},
			}),
			{
				status: 400,
				headers: { "Content-Type": "application/json" },
			},
		);
	}

	return model;
}
