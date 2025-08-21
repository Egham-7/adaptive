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
  {
    selectorId: "sidebar-docs",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Documentation</h3>
        <p className="text-sm text-muted-foreground">
          Access comprehensive guides, API references, and integration examples
          to help you get the most out of Adaptive.
        </p>
      </div>
    ),
    position: "left",
  },
  {
    selectorId: "sidebar-support",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Get Support</h3>
        <p className="text-sm text-muted-foreground">
          Need help? Contact our support team for assistance with setup,
          troubleshooting, or questions about your account.
        </p>
      </div>
    ),
    position: "left",
  },
  {
    selectorId: "sidebar-legal",
    content: (
      <div>
        <h3 className="font-semibold mb-2">Legal Information</h3>
        <p className="text-sm text-muted-foreground">
          Review our terms of service, privacy policy, and other legal documents
          to understand how we protect your data.
        </p>
      </div>
    ),
    position: "left",
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
