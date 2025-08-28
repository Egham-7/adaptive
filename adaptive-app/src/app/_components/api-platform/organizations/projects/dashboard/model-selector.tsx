"use client";

import { useId } from "react";
import MultipleSelector, {
	type Option,
} from "@/components/ui/multiple-selector";
import { api } from "@/trpc/react";

interface ModelSelectorProps {
	selectedModels: string[];
	onModelsChange: (models: string[]) => void;
}

export function ModelSelector({
	selectedModels,
	onModelsChange,
}: ModelSelectorProps) {
	const descriptionId = useId();
	// Fetch all available models from API
	const { data: availableModels, isLoading } =
		api.modelPricing.getAllModelsWithMetadata.useQuery();

	// Convert models to Option format for MultipleSelector
	const modelOptions: Option[] =
		availableModels?.map((model) => ({
			value: `${model.provider.name}:${model.name}`,
			label: model.displayName,
			provider: model.provider.name,
		})) || [];

	// Convert selected model keys back to Option format
	const selectedOptions: Option[] = selectedModels
		.map((modelKey) => modelOptions.find((option) => option.value === modelKey))
		.filter(Boolean) as Option[];

	const handleChange = (options: Option[]) => {
		onModelsChange(options.map((option) => option.value));
	};

	if (isLoading) {
		return <div className="h-10 w-64 animate-pulse rounded-md bg-muted" />;
	}

	return (
		<div className="flex flex-col gap-2">
			<span
				id={descriptionId}
				className="font-medium text-muted-foreground text-sm"
			>
				Compare your Adaptive costs against specific models
			</span>
			<MultipleSelector
				value={selectedOptions}
				onChange={handleChange}
				options={modelOptions}
				inputProps={{
					"aria-label": "Select models to compare",
					"aria-describedby": "model-selector-description",
				}}
				placeholder="Select models to compare..."
				className="w-full"
				maxSelected={5}
				badgeClassName="flex items-center gap-1"
				emptyIndicator={
					<p className="w-full text-center text-gray-600 text-lg leading-10 dark:text-gray-400">
						No models found.
					</p>
				}
			/>
		</div>
	);
}
