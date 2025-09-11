"use client";

import { useEffect, useState } from "react";
import {
  TourAlertDialog,
  TourProvider,
  useTour,
  type TourStep,
} from "@/components/ui/tour";
import { useTourCompletion } from "@/hooks/use-tour-completion";

// Utility function to handle smooth scrolling to tour elements
const scrollToElement = (selectorId: string, fallbackSelector?: string) => {
  try {
    let element = document.getElementById(selectorId);
    
    if (!element && fallbackSelector) {
      element = document.querySelector(fallbackSelector) as HTMLElement;
    }
    
    if (element) {
      element.scrollIntoView({ 
        behavior: "smooth", 
        block: "center",
        inline: "nearest"
      });
      return true;
    }
    
    return false;
  } catch (error) {
    return false;
  }
};

const apiPlatformTourSteps: TourStep[] = [
  {
    selectorId: "team-switcher",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Welcome to Adaptive!</h3>
        <p className="text-sm text-muted-foreground">
          This is your project area. You can navigate back to your organization
          from here.
        </p>
      </div>
    ),
    onBeforeShow: () => scrollToElement("team-switcher"),
  },
  {
    selectorId: "dashboard-header",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Usage Dashboard</h3>
        <p className="text-sm text-muted-foreground">
          Filter your data by date ranges and export reports. Use the date
          picker to view different time periods.
        </p>
      </div>
    ),
    onBeforeShow: () => scrollToElement("dashboard-header"),
  },
  {
    selectorId: "dashboard-metrics",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Key Performance Metrics</h3>
        <p className="text-sm text-muted-foreground">
          Monitor your API spending, savings, request volume, and token usage in
          real-time.
        </p>
      </div>
    ),
    onBeforeShow: () => scrollToElement("dashboard-metrics"),
  },
  {
    selectorId: "dashboard-cost-comparison",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Cost Analysis</h3>
        <p className="text-sm text-muted-foreground">
          Compare costs across different AI providers and models. See how
          Adaptive optimizes your spending.
        </p>
      </div>
    ),
    position: "top",
    onBeforeShow: () => scrollToElement("dashboard-cost-comparison"),
  },
  {
    selectorId: "api-keys-nav",
    content: (
      <div>
        <h3 className="font-semibold mb-2">API Keys</h3>
        <p className="text-sm text-muted-foreground">
          Create your first Adaptive API key
        </p>
      </div>
    ),
    position: "right",
    onBeforeShow: () => scrollToElement("api-keys-nav", '[href*="/api-keys"]'),
  },
];

export function NewUserTour() {
  const { isTourCompleted, setIsTourCompleted, isLoading } =
    useTourCompletion();
  const [showTour, setShowTour] = useState(false);

  const handleTourComplete = async () => {
    await setIsTourCompleted(true);
    setShowTour(false);
  };

  // Show tour when loading is done and tour is not completed
  useEffect(() => {
    if (!isLoading && !isTourCompleted) {
      // Small delay to ensure DOM is ready
      const timer = setTimeout(() => {
        setShowTour(true);
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [isLoading, isTourCompleted]);

  // Don't render anything if still loading
  if (isLoading) {
    return null;
  }

  // Don't render if tour is completed
  if (isTourCompleted) {
    return null;
  }

  return (
    <TourProvider
      onComplete={handleTourComplete}
      isTourCompleted={isTourCompleted}
    >
      <TourContent
        steps={apiPlatformTourSteps}
        showTour={showTour}
        setShowTour={setShowTour}
      />
    </TourProvider>
  );
}

function TourContent({
  steps,
  showTour,
  setShowTour,
}: {
  steps: TourStep[];
  showTour: boolean;
  setShowTour: (show: boolean) => void;
}) {
  const { setSteps } = useTour();

  useEffect(() => {
    setSteps(steps);
  }, [setSteps, steps]);

  return <TourAlertDialog isOpen={showTour} setIsOpen={setShowTour} />;
}
