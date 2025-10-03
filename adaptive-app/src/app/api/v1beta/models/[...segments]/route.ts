import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

export async function POST(
	req: NextRequest,
	{ params }: { params: Promise<{ segments: string[] }> },
) {
	const { segments } = await params;

	// Handle both formats:
	// 1. /models/model-name:generateContent (Gemini colon syntax)
	// 2. /models/model-name/generateContent (standard REST)

	if (segments.length === 1 && segments[0]?.includes(":")) {
		// Handle Gemini colon syntax: "model:method"
		const [model, method] = segments[0].split(":");

		if (method === "generateContent") {
			const { POST: generateContentHandler } = await import(
				"../[model]/generateContent/route"
			);
			return generateContentHandler(req, { params: { model: model || "" } });
		}

		if (method === "streamGenerateContent") {
			const { POST: streamGenerateContentHandler } = await import(
				"../[model]/streamGenerateContent/route"
			);
			return streamGenerateContentHandler(req, {
				params: { model: model || "" },
			});
		}

		if (method === "countTokens") {
			const { POST: countTokensHandler } = await import(
				"../[model]/countTokens/route"
			);
			return countTokensHandler(req, { params: { model: model || "" } });
		}
	}

	if (segments.length === 2) {
		// Handle standard REST format: "model/method"
		const [model, method] = segments;

		if (method === "generateContent") {
			const { POST: generateContentHandler } = await import(
				"../[model]/generateContent/route"
			);
			return generateContentHandler(req, { params: { model: model || "" } });
		}

		if (method === "streamGenerateContent") {
			const { POST: streamGenerateContentHandler } = await import(
				"../[model]/streamGenerateContent/route"
			);
			return streamGenerateContentHandler(req, {
				params: { model: model || "" },
			});
		}

		if (method === "countTokens") {
			const { POST: countTokensHandler } = await import(
				"../[model]/countTokens/route"
			);
			return countTokensHandler(req, { params: { model: model || "" } });
		}
	}

	return new Response(
		JSON.stringify({
			error: {
				code: 404,
				message: "Endpoint not found",
				status: "NOT_FOUND",
			},
		}),
		{
			status: 404,
			headers: { "Content-Type": "application/json" },
		},
	);
}
