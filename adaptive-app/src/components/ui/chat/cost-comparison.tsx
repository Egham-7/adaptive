"use client";

import React, { useState } from "react";
import { DollarSign, TrendingDown, TrendingUp } from "lucide-react";
import Image from "next/image";
import type { UIMessage } from "@ai-sdk/react";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { api } from "@/trpc/react";
import { getProviderLogo } from "@/lib/providers/logos";
import { cn } from "@/lib/shared/utils";

interface CostComparisonProps {
  message: UIMessage;
  selectedModel?: string | null;
  onModelSelect?: (modelId: string) => void;
}

interface CostComparisonResultProps {
  message: UIMessage;
  selectedModel: string | null;
  onReset: () => void;
}

interface ProviderMenuProps {
  provider: { id: string; name: string; displayName: string };
  onModelSelect: (modelId: string) => void;
  onClose?: () => void;
}

const ProviderMenu: React.FC<ProviderMenuProps> = ({
  provider,
  onModelSelect,
  onClose,
}) => {
  const [open, setOpen] = useState(false);
  const {
    data: models,
    isLoading: modelsLoading,
    error: modelsError,
  } = api.modelPricing.getModelsByProvider.useQuery({
    providerId: provider.id,
  });

  const handleModelSelect = (modelId: string) => {
    onModelSelect(modelId);
    setOpen(false); // Close the model popover
    onClose?.(); // Close the main popover
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button className="flex items-center gap-2 w-full p-2 rounded-md hover:bg-accent text-left text-sm">
          {getProviderLogo(provider.name) && (
            <Image
              src={getProviderLogo(provider.name)!}
              alt={provider.name}
              width={16}
              height={16}
              className="rounded-sm"
            />
          )}
          <span>{provider.displayName}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="w-56 p-2"
        side="right"
        align="start"
        sideOffset={8}
      >
        <div className="space-y-2">
          <h5 className="text-sm font-medium">{provider.displayName} Models</h5>
          <div className="space-y-1">
            {models?.map((model) => (
              <button
                key={model.id}
                className="flex items-center justify-between w-full p-2 rounded-md hover:bg-accent text-left text-sm"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleModelSelect(model.id);
                }}
              >
                <span>{model.displayName}</span>
                <span className="text-xs text-muted-foreground">
                  ${Number(model.inputTokenCost)}/${Number(model.outputTokenCost)}/1M
                </span>
              </button>
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export const CostComparison: React.FC<CostComparisonProps> = ({ 
  message, 
  selectedModel, 
  onModelSelect 
}) => {
  // Extract data from message metadata
  const metadata = message.metadata as any;
  const inputTokens = metadata?.usage?.inputTokens || 0;
  const outputTokens = metadata?.usage?.outputTokens || 0;
  const currentProvider = metadata?.providerMetadata?.adaptive?.provider;
  const currentModel = metadata?.response?.modelId;
  const [open, setOpen] = useState(false);

  const { data: providers } = api.modelPricing.getProviders.useQuery();

  const handleModelSelect = (modelId: string) => {
    onModelSelect?.(modelId);
    setOpen(false); // Close the main popover
  };

  // Don't show the component if there's no metadata to compare against
  if (!metadata) {
    return null;
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
          <DollarSign className="h-3 w-3" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 p-2" align="start">
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Compare Costs</h4>
          <div className="text-xs text-muted-foreground space-y-1">
            <p>Comparing against: {currentProvider} {currentModel}</p>
            <p>Usage: {inputTokens.toLocaleString()} input + {outputTokens.toLocaleString()} output tokens</p>
          </div>
          <div className="space-y-1">
            {providers?.map((provider) => (
              <ProviderMenu
                key={provider.id}
                provider={provider}
                onModelSelect={handleModelSelect}
                onClose={() => setOpen(false)}
              />
            ))}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export const CostComparisonResult: React.FC<CostComparisonResultProps> = ({ 
  message, 
  selectedModel, 
  onReset 
}) => {
  // Extract data from message metadata
  const metadata = message.metadata as any;
  const inputTokens = metadata?.usage?.inputTokens || 0;
  const outputTokens = metadata?.usage?.outputTokens || 0;
  const currentProvider = metadata?.providerMetadata?.adaptive?.provider;
  const currentModel = metadata?.response?.modelId;

  const { data: comparison } =
    api.modelPricing.calculateCostComparison.useQuery(
      {
        currentModelName: currentModel || "",
        currentProviderName: currentProvider || "",
        compareModelId: selectedModel!,
        inputTokens,
        outputTokens,
      },
      {
        enabled: !!selectedModel && !!currentModel && !!currentProvider,
      },
    );

  // Don't show the component if there's no metadata to compare against
  if (!metadata || !comparison) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-2 py-1 rounded-md text-xs w-fit",
        comparison.isMoreExpensive
          ? "bg-success/10 text-success border border-success/20"
          : "bg-destructive/10 text-destructive border border-destructive/20",
      )}
    >
      {comparison.isMoreExpensive ? (
        <TrendingDown className="h-3 w-3" />
      ) : (
        <TrendingUp className="h-3 w-3" />
      )}
      <span className="font-medium">
        vs {comparison.compareModel.provider} {comparison.compareModel.name}
      </span>
      <span>
        {comparison.isMoreExpensive ? "Saving $" : "Costs +$"}
        {Math.abs(comparison.savings).toFixed(4)}
      </span>
      <span className="opacity-70">
        ({Math.abs(comparison.savingsPercentage).toFixed(1)}%)
      </span>
      <button
        onClick={onReset}
        className="ml-1 hover:bg-background/20 rounded-sm p-0.5 transition-colors"
      >
        ×
      </button>
    </div>
  );
};

