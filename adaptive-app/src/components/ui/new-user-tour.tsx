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
    position: "auto", // Let the system choose the best position
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
    position: "auto",
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
    position: "auto",
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
  const { isTourCompleted, isLoading } = useTourCompletion();
  const [showTour, setShowTour] = useState(false);

  // Show tour dialog for new users after a short delay
  useEffect(() => {
    if (!isLoading && !isTourCompleted) {
      const timer = setTimeout(() => {
        setShowTour(true);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [isLoading, isTourCompleted]);

  // Don't render anything if tour is completed or still loading
  if (isLoading || isTourCompleted) {
    return null;
  }

  return (
    <TourProvider onComplete={() => setShowTour(false)}>
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

