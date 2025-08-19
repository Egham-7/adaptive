"use client";

import { useEffect, useState } from "react";
import {
  TourAlertDialog,
  TourProvider,
  useTour,
  type TourStep,
} from "@/components/ui/tour";
import { useTourCompletion } from "@/hooks/use-tour-completion";

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

  // Don't render anything if tour is completed or still loading
  if (isLoading || isTourCompleted) {
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
