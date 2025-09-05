import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { authenticateApiKey } from "@/lib/auth-utils";
import { db } from "@/server/db";
import { updateClusterSchema } from "@/types/cluster-schemas";

// GET /api/v1/clusters/{projectId}/{name} - Get cluster by name
export async function GET(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string; name: string }> },
) {
	const { projectId, name } = await params;
	try {
		const apiKey = request.headers.get("authorization")?.replace("Bearer ", "");
		if (!apiKey) {
			return NextResponse.json({ error: "API key required" }, { status: 401 });
		}

		const auth = await authenticateApiKey(apiKey, db);

		// Verify API key has access to this project
		if (auth.apiKey.projectId !== projectId) {
			return NextResponse.json(
				{ error: "API key does not have access to this project" },
				{ status: 403 },
			);
		}

		const cluster = await db.lLMCluster.findFirst({
			where: {
				projectId: projectId,
				name: name,
			},
			include: {
				providers: {
					include: {
						provider: {
							include: {
								models: true,
							},
						},
						config: true,
					},
				},
			},
		});

		if (!cluster) {
			return NextResponse.json({ error: "Cluster not found" }, { status: 404 });
		}

		return NextResponse.json({ cluster });
	} catch (error) {
		console.error("Error fetching cluster:", error);
		return NextResponse.json(
			{ error: "Failed to fetch cluster" },
			{ status: 500 },
		);
	}
}

// PUT /api/v1/clusters/{projectId}/{name} - Update cluster
export async function PUT(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string; name: string }> },
) {
	const { projectId, name } = await params;
	try {
		const apiKey = request.headers.get("authorization")?.replace("Bearer ", "");
		if (!apiKey) {
			return NextResponse.json({ error: "API key required" }, { status: 401 });
		}

		const auth = await authenticateApiKey(apiKey, db);

		// Verify API key has access to this project
		if (auth.apiKey.projectId !== projectId) {
			return NextResponse.json(
				{ error: "API key does not have access to this project" },
				{ status: 403 },
			);
		}

		const rawBody = await request.json();

		// Validate request body with Zod
		const parseResult = updateClusterSchema
			.omit({ id: true, apiKey: true })
			.safeParse(rawBody);
		if (!parseResult.success) {
			return NextResponse.json(
				{ error: "Invalid request", details: parseResult.error.issues },
				{ status: 400 },
			);
		}

		const body = parseResult.data;

		// Find cluster first
		const cluster = await db.lLMCluster.findFirst({
			where: {
				projectId: projectId,
				name: name,
			},
		});

		if (!cluster) {
			return NextResponse.json({ error: "Cluster not found" }, { status: 404 });
		}

		// Update cluster
		const updatedCluster = await db.lLMCluster.update({
			where: { id: cluster.id },
			data: {
				...(body.description !== undefined && {
					description: body.description,
				}),
				...(body.fallbackEnabled !== undefined && {
					fallbackEnabled: body.fallbackEnabled,
				}),
				...(body.fallbackMode && { fallbackMode: body.fallbackMode }),
				...(body.enableCircuitBreaker !== undefined && {
					enableCircuitBreaker: body.enableCircuitBreaker,
				}),
				...(body.maxRetries !== undefined && { maxRetries: body.maxRetries }),
				...(body.timeoutMs !== undefined && { timeoutMs: body.timeoutMs }),
				...(body.costBias !== undefined && { costBias: body.costBias }),
				...(body.complexityThreshold !== undefined && {
					complexityThreshold: body.complexityThreshold,
				}),
				...(body.tokenThreshold !== undefined && {
					tokenThreshold: body.tokenThreshold,
				}),
				...(body.enableSemanticCache !== undefined && {
					enableSemanticCache: body.enableSemanticCache,
				}),
				...(body.semanticThreshold !== undefined && {
					semanticThreshold: body.semanticThreshold,
				}),
				...(body.enablePromptCache !== undefined && {
					enablePromptCache: body.enablePromptCache,
				}),
				...(body.promptCacheTTL !== undefined && {
					promptCacheTTL: body.promptCacheTTL,
				}),
				// Note: isActive field was removed from schema
			},
			include: {
				providers: {
					include: {
						provider: {
							include: {
								models: true,
							},
						},
						config: true,
					},
				},
			},
		});

		return NextResponse.json({ cluster: updatedCluster });
	} catch (error) {
		console.error("Error updating cluster:", error);
		return NextResponse.json(
			{ error: "Failed to update cluster" },
			{ status: 500 },
		);
	}
}

// DELETE /api/v1/clusters/{projectId}/{name} - Delete cluster
export async function DELETE(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string; name: string }> },
) {
	const { projectId, name } = await params;
	try {
		const apiKey = request.headers.get("authorization")?.replace("Bearer ", "");
		if (!apiKey) {
			return NextResponse.json({ error: "API key required" }, { status: 401 });
		}

		const auth = await authenticateApiKey(apiKey, db);

		// Verify API key has access to this project
		if (auth.apiKey.projectId !== projectId) {
			return NextResponse.json(
				{ error: "API key does not have access to this project" },
				{ status: 403 },
			);
		}

		// Find cluster first
		const cluster = await db.lLMCluster.findFirst({
			where: {
				projectId: projectId,
				name: name,
			},
		});

		if (!cluster) {
			return NextResponse.json({ error: "Cluster not found" }, { status: 404 });
		}

		// Hard delete (since isActive was removed from schema)
		await db.lLMCluster.delete({
			where: { id: cluster.id },
		});

		return NextResponse.json({ success: true });
	} catch (error) {
		console.error("Error deleting cluster:", error);
		return NextResponse.json(
			{ error: "Failed to delete cluster" },
			{ status: 500 },
		);
	}
}
