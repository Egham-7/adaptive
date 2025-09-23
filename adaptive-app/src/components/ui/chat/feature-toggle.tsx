import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/shared/utils";

interface FeatureToggleProps {
  icon: LucideIcon;
  label: string;
  isEnabled: boolean;
  onToggle: () => void;
  ariaLabel: string;
  className?: string;
}

export function FeatureToggle({
  icon: Icon,
  label,
  isEnabled,
  onToggle,
  ariaLabel,
  className,
}: FeatureToggleProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={ariaLabel}
      aria-pressed={isEnabled}
      className={cn(
        "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
        isEnabled
          ? "bg-primary/10 text-primary hover:bg-primary/20"
          : "bg-muted text-muted-foreground hover:bg-muted/80",
        className
      )}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
    </button>
  );
}