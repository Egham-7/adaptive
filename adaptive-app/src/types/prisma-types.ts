import type { Prisma } from "../../prisma/generated";

// Simple reusable Prisma payload types for MVP cluster functionality

export type ProviderWithModels = Prisma.ProviderGetPayload<{
	include: {
		models: {
			include: {
				capabilities: true;
			};
		};
	};
}>;

export type ClusterWithProviders = Prisma.LLMClusterGetPayload<{
	include: {
		providers: {
			include: {
				provider: {
					include: {
						models: {
							include: {
								capabilities: true;
							};
						};
					};
				};
				config: true;
			};
		};
	};
}>;

export type ClusterProviderWithModels = Prisma.ClusterProviderGetPayload<{
	include: {
		provider: {
			include: {
				models: {
					include: {
						capabilities: true;
					};
				};
			};
		};
		config: true;
	};
}>;
