import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { authenticateApiKey } from "@/lib/auth-utils";
import { db } from "@/server/db";

// POST /api/v1/clusters/{projectId}/{name}/chat/completions - OpenAI-compatible chat completions with cluster routing
export async function POST(
	request: NextRequest,
	{ params }: { params: { projectId: string; name: string } },
) {
	try {
		const apiKey = request.headers.get("authorization")?.replace("Bearer ", "");
		if (!apiKey) {
			return NextResponse.json({ error: "API key required" }, { status: 401 });
		}

		const auth = await authenticateApiKey(apiKey, db);

		// Verify API key has access to this project
		if (auth.apiKey.projectId !== params.projectId) {
			return NextResponse.json(
				{ error: "API key does not have access to this project" },
				{ status: 403 },
			);
		}

		// Get cluster configuration
		const cluster = await db.lLMCluster.findFirst({
			where: {
				projectId: params.projectId,
				name: params.name,
				isActive: true,
			},
			include: {
				models: {
					where: { isActive: true },
					orderBy: { priority: "asc" },
				},
			},
		});

		if (!cluster) {
			return NextResponse.json({ error: "Cluster not found" }, { status: 404 });
		}

		const body = await request.json();

		// Build the enhanced chat completion request with cluster config
		const enhancedRequest = {
			...body,
			user: params.name, // Pass cluster name in user field for backend routing
			protocol_manager: {
				models: cluster.models.map((m) => ({
					provider: m.provider,
					model_name: m.modelName,
					cost_per_1m_input_tokens: 0, // Will be filled by backend
					cost_per_1m_output_tokens: 0,
					max_context_tokens: 0,
					supports_function_calling: true,
				})),
				cost_bias: cluster.costBias,
				complexity_threshold: cluster.complexityThreshold,
				token_threshold: cluster.tokenThreshold,
			},
			semantic_cache: {
				enabled: cluster.enableSemanticCache,
				semantic_threshold: cluster.semanticThreshold,
			},
			prompt_cache: {
				enabled: cluster.enablePromptCache,
				ttl: cluster.promptCacheTTL,
			},
			fallback: {
				enabled: cluster.fallbackEnabled,
				mode: cluster.fallbackMode,
			},
		};

		// Forward to backend Go API
		const backendUrl =
			process.env.NEXT_PUBLIC_ADAPTIVE_API_URL || "http://localhost:8080";
		const response = await fetch(`${backendUrl}/v1/chat/completions`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${apiKey}`,
			},
			body: JSON.stringify(enhancedRequest),
		});

		if (!response.ok) {
			const errorData = await response
				.json()
				.catch(() => ({ error: "Unknown error" }));
			return NextResponse.json(
				{ error: errorData.error || "Backend request failed" },
				{ status: response.status },
			);
		}

		// Handle streaming response
		if (body.stream) {
			return new Response(response.body, {
				status: response.status,
				headers: {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				},
			});
		}

		const data = await response.json();
		return NextResponse.json(data);
	} catch (error) {
		console.error("Error calling cluster chat completion:", error);
		return NextResponse.json(
			{ error: "Failed to process chat completion" },
			{ status: 500 },
		);
	}
}
