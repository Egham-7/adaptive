"use client";

import { Check, ChevronsUpDown, Search } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Command,
	CommandEmpty,
	CommandGroup,
	CommandInput,
	CommandItem,
	CommandList,
} from "@/components/ui/command";
import {
	Popover,
	PopoverContent,
	PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { api } from "@/trpc/react";

interface ModelSelectorProps {
	selectedModel: string;
	onModelChange: (model: string) => void;
}

// Popular models (could be fetched from API based on usage stats)
const POPULAR_MODELS = [
	{ id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
	{ id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI" },
	{ id: "claude-3.5-sonnet", name: "Claude 3.5 Sonnet", provider: "Anthropic" },
	{ id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "Google" },
	{ id: "deepseek-chat", name: "DeepSeek Chat", provider: "DeepSeek" },
];

export function ModelSelector({
	selectedModel,
	onModelChange,
}: ModelSelectorProps) {
	const [open, setOpen] = useState(false);
	const [searchValue, setSearchValue] = useState("");

	// Fetch all available models from API
	const { data: allModels, isLoading } =
		api.modelPricing.getAllModelPricing.useQuery();

	// Convert pricing data to model list format
	const availableModels = allModels
		? Object.keys(allModels).map((modelId) => {
				// Try to find provider from popular models first
				const popularModel = POPULAR_MODELS.find((m) => m.id === modelId);
				if (popularModel) {
					return popularModel;
				}

				// Infer provider from model ID
				let provider = "Others";
				if (modelId.includes("gpt")) provider = "OpenAI";
				else if (modelId.includes("claude")) provider = "Anthropic";
				else if (modelId.includes("gemini")) provider = "Google";
				else if (modelId.includes("deepseek")) provider = "DeepSeek";
				else if (modelId.includes("llama")) provider = "Meta";

				return {
					id: modelId,
					name:
						modelId.charAt(0).toUpperCase() +
						modelId.slice(1).replace(/-/g, " "),
					provider,
				};
			})
		: [];

	// Filter models based on search
	const filteredModels = availableModels.filter(
		(model) =>
			model.name.toLowerCase().includes(searchValue.toLowerCase()) ||
			model.provider.toLowerCase().includes(searchValue.toLowerCase()) ||
			model.id.toLowerCase().includes(searchValue.toLowerCase()),
	);

	// Separate popular and other models
	const popularModelsFiltered = filteredModels.filter((model) =>
		POPULAR_MODELS.some((popular) => popular.id === model.id),
	);

	const otherModelsFiltered = filteredModels.filter(
		(model) => !POPULAR_MODELS.some((popular) => popular.id === model.id),
	);

	// Group other models by provider
	const modelsByProvider = otherModelsFiltered.reduce<
		Record<string, typeof otherModelsFiltered>
	>((acc, model) => {
		if (!acc[model.provider]) {
			acc[model.provider] = [];
		}
		acc[model.provider]?.push(model);
		return acc;
	}, {});

	const selectedModelInfo = availableModels.find((m) => m.id === selectedModel);

	return (
		<div className="flex items-center gap-2">
			<span className="font-medium text-muted-foreground text-sm">
				Compare with:
			</span>
			<Popover open={open} onOpenChange={setOpen}>
				<PopoverTrigger asChild>
					{/* biome-ignore lint/a11y/useSemanticElements: This is a proper combobox implementation using ARIA patterns */}
					<Button
						variant="outline"
						role="combobox"
						aria-expanded={open}
						className="w-64 justify-between"
						disabled={isLoading}
					>
						{selectedModelInfo ? (
							<div className="flex items-center gap-2">
								<span className="font-medium">{selectedModelInfo.name}</span>
								<span className="text-muted-foreground text-xs">
									{selectedModelInfo.provider}
								</span>
							</div>
						) : (
							"Select model..."
						)}
						<ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
					</Button>
				</PopoverTrigger>
				<PopoverContent className="w-80 p-0" align="start">
					<Command>
						<div className="flex items-center border-b px-3">
							<Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
							<CommandInput
								placeholder="Search models..."
								value={searchValue}
								onValueChange={setSearchValue}
								className="border-0 focus:ring-0"
							/>
						</div>
						<CommandList className="max-h-80">
							<CommandEmpty>No models found.</CommandEmpty>

							{/* Popular Models */}
							{popularModelsFiltered.length > 0 && (
								<CommandGroup heading="Popular Models">
									{popularModelsFiltered.map((model) => (
										<CommandItem
											key={model.id}
											value={model.id}
											onSelect={() => {
												onModelChange(model.id);
												setOpen(false);
											}}
											className="flex items-center justify-between"
										>
											<div className="flex flex-col">
												<span className="font-medium">{model.name}</span>
												<span className="text-muted-foreground text-xs">
													{model.provider}
												</span>
											</div>
											<Check
												className={cn(
													"h-4 w-4",
													selectedModel === model.id
														? "opacity-100"
														: "opacity-0",
												)}
											/>
										</CommandItem>
									))}
								</CommandGroup>
							)}

							{/* All Models by Provider */}
							{Object.entries(modelsByProvider).map(([provider, models]) => (
								<CommandGroup key={provider} heading={provider}>
									{models.map((model) => (
										<CommandItem
											key={model.id}
											value={model.id}
											onSelect={() => {
												onModelChange(model.id);
												setOpen(false);
											}}
											className="flex items-center justify-between"
										>
											<div className="flex flex-col">
												<span className="font-medium">{model.name}</span>
												<span className="text-muted-foreground text-xs">
													{model.id}
												</span>
											</div>
											<Check
												className={cn(
													"h-4 w-4",
													selectedModel === model.id
														? "opacity-100"
														: "opacity-0",
												)}
											/>
										</CommandItem>
									))}
								</CommandGroup>
							))}
						</CommandList>
					</Command>
				</PopoverContent>
			</Popover>
		</div>
	);
}
