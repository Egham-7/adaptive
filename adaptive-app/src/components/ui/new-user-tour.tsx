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
    selectorId: "dashboard-header",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Welcome to Adaptive!</h3>
        <p className="text-sm text-muted-foreground">
          This is your usage dashboard. Monitor your API spending, savings, request volume, and token usage in real-time. Use the date picker to filter data by different time periods and export reports.
        </p>
      </div>
    ),
    onBeforeShow: () => scrollToElement("dashboard-header"),
  },
  {
    selectorId: "api-keys-nav",
    content: (
      <div>
        <h3 className="font-semibold mb-2">API Keys</h3>
        <p className="text-sm text-muted-foreground">
          Click here to navigate to the API Keys page where you can manage your keys.
        </p>
      </div>
    ),
    position: "right",
    onBeforeShow: () => scrollToElement("api-keys-nav", '[href*="/api-keys"]'),
  },
  {
    selectorId: "create-api-key-button",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Create Your First API Key</h3>
        <p className="text-sm text-muted-foreground">
          Click this button to create your first Adaptive API key and start integrating with your applications.
        </p>
      </div>
    ),
    position: "bottom",
    onBeforeShow: () => scrollToElement("create-api-key-button"),
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
