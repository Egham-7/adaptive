import { type NextRequest, NextResponse } from "next/server";
import { authenticateApiKey } from "@/lib/auth-utils";
import { db } from "@/server/db";
import { createClusterSchema } from "@/types/cluster-schemas";

// GET /api/v1/clusters/{projectId} - List all clusters for project
export async function GET(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string }> },
) {
	const { projectId } = await params;
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

		const clusters = await db.lLMCluster.findMany({
			where: {
				projectId: projectId,
				isActive: true,
			},
			include: {
				models: {
					where: { isActive: true },
					orderBy: { priority: "asc" },
				},
			},
			orderBy: { createdAt: "desc" },
		});

		return NextResponse.json({ clusters });
	} catch (error) {
		console.error("Error fetching clusters:", error);
		return NextResponse.json(
			{ error: "Failed to fetch clusters" },
			{ status: 500 },
		);
	}
}

// POST /api/v1/clusters/{projectId} - Create new cluster
export async function POST(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string }> },
) {
	const { projectId } = await params;
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
		const parseResult = createClusterSchema
			.omit({ projectId: true, apiKey: true })
			.safeParse(rawBody);
		if (!parseResult.success) {
			return NextResponse.json(
				{ error: "Invalid request", details: parseResult.error.issues },
				{ status: 400 },
			);
		}

		const body = parseResult.data;

		// Create cluster with transaction
		const cluster = await db.$transaction(async (tx) => {
			// Check if cluster name already exists
			const existing = await tx.lLMCluster.findFirst({
				where: {
					projectId: projectId,
					name: body.name,
				},
			});

			if (existing) {
				throw new Error("Cluster name already exists in this project");
			}

			// Validate all models exist
			for (const model of body.models) {
				const providerModel = await tx.providerModel.findFirst({
					where: {
						provider: { name: model.provider },
						name: model.modelName,
						isActive: true,
					},
				});

				if (!providerModel) {
					throw new Error(
						`Model ${model.modelName} from ${model.provider} not found`,
					);
				}
			}

			// Create cluster
			const newCluster = await tx.lLMCluster.create({
				data: {
					projectId: projectId,
					name: body.name,
					description: body.description,
					fallbackEnabled: body.fallbackEnabled ?? true,
					fallbackMode: body.fallbackMode ?? "parallel",
					enableCircuitBreaker: body.enableCircuitBreaker ?? true,
					maxRetries: body.maxRetries ?? 3,
					timeoutMs: body.timeoutMs ?? 30000,
					costBias: body.costBias ?? 0.5,
					complexityThreshold: body.complexityThreshold,
					tokenThreshold: body.tokenThreshold,
					enableSemanticCache: body.enableSemanticCache ?? true,
					semanticThreshold: body.semanticThreshold ?? 0.85,
					enablePromptCache: body.enablePromptCache ?? true,
					promptCacheTTL: body.promptCacheTTL ?? 3600,
				},
			});

			// Create cluster models
			await tx.clusterModel.createMany({
				data: body.models.map((model) => ({
					clusterId: newCluster.id,
					provider: model.provider,
					modelName: model.modelName,
					priority: model.priority ?? 1,
				})),
			});

			// Return cluster with models
			return await tx.lLMCluster.findUnique({
				where: { id: newCluster.id },
				include: {
					models: {
						orderBy: { priority: "asc" },
					},
				},
			});
		});

		return NextResponse.json({ cluster });
	} catch (error) {
		console.error("Error creating cluster:", error);
		return NextResponse.json(
			{
				error:
					error instanceof Error ? error.message : "Failed to create cluster",
			},
			{ status: 500 },
		);
	}
}
