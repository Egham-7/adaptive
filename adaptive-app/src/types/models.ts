import type { RouterOutputs } from "@/trpc/react";

// Model types inferred from tRPC router outputs
export type ModelWithMetadata =
	RouterOutputs["modelPricing"]["getAllModelsWithMetadata"][number];
export type ModelsByProvider =
	RouterOutputs["modelPricing"]["getModelsByProvider"];
export type CostComparison =
	RouterOutputs["modelPricing"]["calculateCostComparison"];
