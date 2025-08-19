"use client";

import { Check, ChevronsUpDown } from "lucide-react";
import Image from "next/image";
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
import { getProviderLogo } from "@/lib/providers";
import { cn } from "@/lib/utils";
import { api } from "@/trpc/react";
import type { ModelWithMetadata } from "@/types/models";

interface ModelSelectorProps {
	selectedModel: string;
	onModelChange: (model: string) => void;
}

export function ModelSelector({
	selectedModel,
	onModelChange,
}: ModelSelectorProps) {
	const [open, setOpen] = useState(false);
	const [searchValue, setSearchValue] = useState("");

	// Fetch all available models from API
	const { data: availableModels, isLoading } =
		api.modelPricing.getAllModelsWithMetadata.useQuery();

	// Filter models based on search
	const filteredModels =
		availableModels?.filter(
			(model: ModelWithMetadata) =>
				model.displayName.toLowerCase().includes(searchValue.toLowerCase()) ||
				model.provider.name.toLowerCase().includes(searchValue.toLowerCase()) ||
				model.id.toLowerCase().includes(searchValue.toLowerCase()),
		) || [];

	const selectedModelInfo = availableModels?.find(
		(m: ModelWithMetadata) => m.id === selectedModel,
	);

	return (
		<div className="flex items-center gap-2">
			<span className="font-medium text-muted-foreground text-sm">
				Compare with:
			</span>
			<Popover open={open} onOpenChange={setOpen}>
				<PopoverTrigger asChild>
					<Button
						variant="outline"
						aria-expanded={open}
						className="w-64 justify-between"
						disabled={isLoading}
					>
						{selectedModelInfo ? (
							<div className="flex items-center gap-2">
								{(() => {
									const logoSrc = getProviderLogo(
										selectedModelInfo.provider.name.toLowerCase(),
									);
									return logoSrc ? (
										<Image
											src={logoSrc}
											alt={selectedModelInfo.provider.name}
											width={16}
											height={16}
											className="rounded-sm"
										/>
									) : null;
								})()}
								<span className="font-medium">
									{selectedModelInfo.displayName}
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
						<CommandInput
							placeholder="Search models..."
							value={searchValue}
							onValueChange={setSearchValue}
							className="border-0 focus:ring-0"
						/>
						<CommandList className="max-h-80">
							<CommandEmpty>No models found.</CommandEmpty>
							<CommandGroup>
								{filteredModels.map((model) => (
									<CommandItem
										key={model.id}
										value={model.id}
										onSelect={() => {
											onModelChange(model.id);
											setOpen(false);
										}}
										className="flex items-center justify-between"
									>
										<div className="flex items-center gap-2">
											{(() => {
												const logoSrc = getProviderLogo(
													model.provider.name.toLowerCase(),
												);
												return logoSrc ? (
													<Image
														src={logoSrc}
														alt={model.provider.name}
														width={16}
														height={16}
														className="rounded-sm"
													/>
												) : null;
											})()}
											<span className="font-medium">{model.displayName}</span>
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
						</CommandList>
					</Command>
				</PopoverContent>
			</Popover>
		</div>
	);
}
