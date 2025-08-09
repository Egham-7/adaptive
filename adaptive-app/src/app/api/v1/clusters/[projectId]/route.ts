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
				return NextResponse.json(
					{ error: "Cluster name already exists in this project" },
					{ status: 409 },
				);
			}

			// Validate all providers and models exist and are accessible
			for (const providerInput of body.providers) {
				// Check if provider exists and is accessible to this project
				const provider = await tx.provider.findFirst({
					where: {
						id: providerInput.providerId,
						OR: [
							{ visibility: "system" },
							{ visibility: "community" },
							{ projectId: projectId },
							// TODO: Add organization-scoped providers
						],
					},
					include: {
						models: {
							include: {
								capabilities: true,
							},
						},
					},
				});

				if (!provider) {
					return NextResponse.json(
						{ error: "Provider not found or not accessible" },
						{ status: 404 },
					);
				}

				// Note: Model selection is now handled via the ProviderConfigModel junction table
				// The new schema uses selectedModels relationship (empty = use all models)

				// If configId is specified, validate it exists and belongs to this project
				if (providerInput.configId) {
					const providerConfig = await tx.providerConfig.findFirst({
						where: {
							id: providerInput.configId,
							projectId: projectId,
							providerId: providerInput.providerId,
						},
					});

					if (!providerConfig) {
						return NextResponse.json(
							{ error: "Provider config not found or not accessible" },
							{ status: 404 },
						);
					}
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

			// Create cluster providers
			await tx.clusterProvider.createMany({
				data: body.providers.map((provider) => ({
					clusterId: newCluster.id,
					providerId: provider.providerId,
					configId: provider.configId,
				})),
			});

			// Return cluster with providers
			return await tx.lLMCluster.findUnique({
				where: { id: newCluster.id },
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
