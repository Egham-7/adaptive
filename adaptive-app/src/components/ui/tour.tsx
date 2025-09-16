import { AnimatePresence, motion } from "motion/react";
import type React from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { Torus } from "lucide-react";

export interface TourStep {
  content: React.ReactNode;
  selectorId: string;
  width?: number;
  height?: number;
  onClickWithinArea?: () => void;
  position?: "top" | "bottom" | "left" | "right";
  onBeforeShow?: () => void;
}

interface TourContextType {
  currentStep: number;
  totalSteps: number;
  nextStep: () => void;
  previousStep: () => void;
  endTour: () => void;
  skipTour: () => void;
  isActive: boolean;
  startTour: () => void;
  setSteps: (steps: TourStep[]) => void;
  steps: TourStep[];
  isTourCompleted: boolean;
  setIsTourCompleted: (completed: boolean) => void;
  onComplete?: () => void;
}

interface TourProviderProps {
  children: React.ReactNode;
  onComplete?: () => void;
  className?: string;
  isTourCompleted?: boolean;
}

const TourContext = createContext<TourContextType | null>(null);

const PADDING = 16;
const MIN_CONTENT_WIDTH = 280;
const MAX_CONTENT_WIDTH = 420;

function getElementPosition(id: string) {
  const element = document.getElementById(id);
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top,
    left: rect.left,
    width: rect.width,
    height: rect.height,
  };
}

function calculateContentPosition(
  elementPos: { top: number; left: number; width: number; height: number },
  position: "top" | "bottom" | "left" | "right" = "bottom",
  contentWidth: number = 320,
  contentHeight: number = 200,
) {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  
  // Ensure content fits within viewport with padding
  const maxAllowedWidth = viewportWidth - PADDING * 4;
  const maxAllowedHeight = viewportHeight - PADDING * 4;
  
  const CONTENT_WIDTH = Math.min(
    Math.max(contentWidth, MIN_CONTENT_WIDTH),
    Math.min(MAX_CONTENT_WIDTH, maxAllowedWidth)
  );
  const CONTENT_HEIGHT = Math.min(contentHeight, maxAllowedHeight);

  // Smart positioning: try positions in order of preference
  const positions = [
    position, // User preference first
    ...(["bottom", "top", "right", "left"] as const).filter(p => p !== position)
  ];

  for (const pos of positions) {
    let left = elementPos.left;
    let top = elementPos.top;

    switch (pos) {
      case "top":
        top = elementPos.top - CONTENT_HEIGHT - PADDING;
        left = elementPos.left + elementPos.width / 2 - CONTENT_WIDTH / 2;
        break;
      case "bottom":
        top = elementPos.top + elementPos.height + PADDING;
        left = elementPos.left + elementPos.width / 2 - CONTENT_WIDTH / 2;
        break;
      case "left":
        left = elementPos.left - CONTENT_WIDTH - PADDING;
        top = elementPos.top + elementPos.height / 2 - CONTENT_HEIGHT / 2;
        break;
      case "right":
        left = elementPos.left + elementPos.width + PADDING;
        top = elementPos.top + elementPos.height / 2 - CONTENT_HEIGHT / 2;
        break;
    }

    // Check if this position fits in viewport
    const fitsHorizontally = left >= scrollX + PADDING && left + CONTENT_WIDTH <= scrollX + viewportWidth - PADDING;
    const fitsVertically = top >= scrollY + PADDING && top + CONTENT_HEIGHT <= scrollY + viewportHeight - PADDING;

    if (fitsHorizontally && fitsVertically) {
      return { top, left, width: CONTENT_WIDTH, height: CONTENT_HEIGHT };
    }
  }

  // Fallback: position with viewport constraints
  let left = Math.max(scrollX + PADDING, Math.min(
    elementPos.left + elementPos.width / 2 - CONTENT_WIDTH / 2,
    scrollX + viewportWidth - CONTENT_WIDTH - PADDING
  ));
  
  let top = Math.max(scrollY + PADDING, Math.min(
    elementPos.top + elementPos.height + PADDING,
    scrollY + viewportHeight - CONTENT_HEIGHT - PADDING
  ));

  return { top, left, width: CONTENT_WIDTH, height: CONTENT_HEIGHT };
}

export function TourProvider({
  children,
  onComplete,
  className,
  isTourCompleted = false,
}: TourProviderProps) {
  const [steps, setSteps] = useState<TourStep[]>([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [elementPosition, setElementPosition] = useState<{
    top: number;
    left: number;
    width: number;
    height: number;
  } | null>(null);
  const [contentDimensions, setContentDimensions] = useState({ width: 320, height: 200 });
  const [isCompleted, setIsCompleted] = useState(isTourCompleted);
  const contentRef = useRef<HTMLDivElement>(null);

  const updateElementPosition = useCallback(() => {
    if (currentStep >= 0 && currentStep < steps.length) {
      const step = steps[currentStep];
      if (!step) return;

      const position = getElementPosition(step.selectorId ?? "");
      if (position) {
        setElementPosition(position);
      }
    }
  }, [currentStep, steps]);

  useEffect(() => {
    updateElementPosition();
    window.addEventListener("resize", updateElementPosition);
    window.addEventListener("scroll", updateElementPosition);

    return () => {
      window.removeEventListener("resize", updateElementPosition);
      window.removeEventListener("scroll", updateElementPosition);
    };
  }, [updateElementPosition]);

  // Separate effect to handle step changes and call onBeforeShow
  useEffect(() => {
    if (currentStep >= 0 && currentStep < steps.length) {
      const step = steps[currentStep];
      if (step?.onBeforeShow) {
        step.onBeforeShow();
      }
    }
  }, [currentStep, steps]);

  // Measure content dimensions when step changes
  useEffect(() => {
    if (contentRef.current && currentStep >= 0) {
      const measureContent = () => {
        if (contentRef.current) {
          const rect = contentRef.current.getBoundingClientRect();
          setContentDimensions({
            width: Math.max(rect.width, MIN_CONTENT_WIDTH),
            height: Math.max(rect.height, 150)
          });
        }
      };

      // Measure after content has rendered
      const timer = setTimeout(measureContent, 50);
      return () => clearTimeout(timer);
    }
  }, [currentStep, steps]);

  const setIsTourCompleted = useCallback((completed: boolean) => {
    setIsCompleted(completed);
  }, []);

  const nextStep = useCallback(async () => {
    const isLastStep = currentStep >= steps.length - 1;

    setCurrentStep((prev) => {
      if (prev >= steps.length - 1) {
        return -1;
      }
      return prev + 1;
    });

    // Handle completion after state update
    if (isLastStep) {
      setIsTourCompleted(true);
      onComplete?.();
    }
  }, [currentStep, steps.length, onComplete, setIsTourCompleted]);

  const previousStep = useCallback(() => {
    setCurrentStep((prev) => (prev > 0 ? prev - 1 : prev));
  }, []);

  const endTour = useCallback(() => {
    setCurrentStep(-1);
  }, []);

  const skipTour = useCallback(() => {
    setCurrentStep(-1);
    setIsTourCompleted(true);
    onComplete?.();
  }, [setIsTourCompleted, onComplete]);

  const startTour = useCallback(() => {
    if (isTourCompleted) {
      return;
    }
    setCurrentStep(0);
  }, [isTourCompleted]);

  const handleClick = useCallback(
    (e: MouseEvent) => {
      if (
        currentStep >= 0 &&
        elementPosition &&
        steps[currentStep]?.onClickWithinArea
      ) {
        const clickX = e.clientX + window.scrollX;
        const clickY = e.clientY + window.scrollY;

        const isWithinBounds =
          clickX >= elementPosition.left &&
          clickX <=
            elementPosition.left +
              (steps[currentStep]?.width || elementPosition.width) &&
          clickY >= elementPosition.top &&
          clickY <=
            elementPosition.top +
              (steps[currentStep]?.height || elementPosition.height);

        if (isWithinBounds) {
          steps[currentStep].onClickWithinArea?.();
        }
      }
    },
    [currentStep, elementPosition, steps],
  );

  useEffect(() => {
    window.addEventListener("click", handleClick);
    return () => {
      window.removeEventListener("click", handleClick);
    };
  }, [handleClick]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (currentStep < 0) return;
      
      switch (e.key) {
        case "Escape":
          e.preventDefault();
          endTour();
          break;
        case "ArrowRight":
        case " ":
          e.preventDefault();
          nextStep();
          break;
        case "ArrowLeft":
          e.preventDefault();
          if (currentStep > 0) previousStep();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentStep, nextStep, previousStep, endTour]);

  return (
    <TourContext.Provider
      value={{
        currentStep,
        totalSteps: steps.length,
        nextStep,
        previousStep,
        endTour,
        skipTour,
        isActive: currentStep >= 0,
        startTour,
        setSteps,
        steps,
        isTourCompleted: isCompleted,
        setIsTourCompleted,
        onComplete,
      }}
    >
      {children}
      <AnimatePresence>
        {currentStep >= 0 && elementPosition && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 overflow-hidden bg-black/50"
              style={{
                clipPath: `polygon(
                  0% 0%,
                  0% 100%,
                  100% 100%,
                  100% 0%,
                  
                  ${elementPosition.left}px 0%,
                  ${elementPosition.left}px ${elementPosition.top}px,
                  ${elementPosition.left + (steps[currentStep]?.width || elementPosition.width)}px ${elementPosition.top}px,
                  ${elementPosition.left + (steps[currentStep]?.width || elementPosition.width)}px ${elementPosition.top + (steps[currentStep]?.height || elementPosition.height)}px,
                  ${elementPosition.left}px ${elementPosition.top + (steps[currentStep]?.height || elementPosition.height)}px,
                  ${elementPosition.left}px 0%
                )`,
              }}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              style={{
                position: "fixed",
                top: elementPosition.top,
                left: elementPosition.left,
                width: elementPosition.width,
                height: elementPosition.height,
              }}
              className={cn("z-[100] border-2 border-primary", className)}
            />

            <motion.div
              ref={contentRef}
              initial={{ opacity: 0, y: 10 }}
              animate={{
                opacity: 1,
                y: 0,
                ...calculateContentPosition(
                  elementPosition,
                  steps[currentStep]?.position,
                  contentDimensions.width,
                  contentDimensions.height,
                ),
              }}
              transition={{
                duration: 0.3,
                ease: "easeOut",
              }}
              exit={{ opacity: 0, y: 10 }}
              style={{
                position: "fixed",
                zIndex: 1000,
              }}
              className="bg-background relative rounded-xl border shadow-xl w-auto"
            >
              {/* Header with progress and close */}
              <div className="flex items-center justify-between p-4 pb-2">
                <div className="flex items-center gap-2">
                  {/* Progress dots */}
                  <div className="flex gap-1">
                    {Array.from({ length: steps.length }, (_, i) => (
                      <div
                        key={i}
                        className={cn(
                          "h-1.5 w-1.5 rounded-full transition-colors",
                          i === currentStep
                            ? "bg-primary"
                            : i < currentStep
                            ? "bg-primary/60"
                            : "bg-muted"
                        )}
                      />
                    ))}
                  </div>
                  <span className="text-muted-foreground text-xs ml-1">
                    {currentStep + 1} of {steps.length}
                  </span>
                </div>
                <Button
                  onClick={endTour}
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                >
                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </Button>
              </div>

              {/* Content */}
              <div className="px-4 pb-4">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={`tour-content-${currentStep}`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                    {steps[currentStep]?.content}
                  </motion.div>
                </AnimatePresence>

                {/* Actions */}
                <div className="flex items-center justify-between mt-6 pt-4 border-t">
                  <div className="flex gap-2">
                    {currentStep > 0 && (
                      <Button
                        onClick={previousStep}
                        variant="outline"
                        size="sm"
                        className="text-xs h-8 px-3"
                      >
                        Back
                      </Button>
                    )}
                    <Button
                      onClick={skipTour}
                      variant="ghost"
                      size="sm"
                      className="text-xs h-8 px-3 text-muted-foreground"
                    >
                      Skip Tour
                    </Button>
                  </div>
                  <Button
                    onClick={nextStep}
                    size="sm"
                    className="text-xs h-8 px-4 font-medium"
                  >
                    {currentStep === steps.length - 1 ? "Finish" : "Next"}
                  </Button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </TourContext.Provider>
  );
}

export function useTour() {
  const context = useContext(TourContext);
  if (!context) {
    throw new Error("useTour must be used within a TourProvider");
  }
  return context;
}

export function TourAlertDialog({
  isOpen,
  setIsOpen,
}: {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}) {
  const { startTour, steps, isTourCompleted, currentStep, setIsTourCompleted, onComplete } = useTour();

  if (isTourCompleted || steps.length === 0 || currentStep > -1) {
    return null;
  }

  const handleSkip = async () => {
    setIsOpen(false);
    setIsTourCompleted(true);
    onComplete?.();
  };

  return (
    <AlertDialog open={isOpen}>
      <AlertDialogContent className="max-w-md p-6">
        <AlertDialogHeader className="flex flex-col items-center justify-center">
          <div className="relative mb-4">
            <motion.div
              initial={{ scale: 0.7, filter: "blur(10px)" }}
              animate={{
                scale: 1,
                filter: "blur(0px)",
                y: [0, -8, 0],
                rotate: [42, 48, 42],
              }}
              transition={{
                duration: 0.4,
                ease: "easeOut",
                y: {
                  duration: 2.5,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "easeInOut",
                },
                rotate: {
                  duration: 3,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "easeInOut",
                },
              }}
            >
              <Torus className="size-32 stroke-1 text-primary" />
            </motion.div>
          </div>
          <AlertDialogTitle className="text-center text-xl font-medium">
            Welcome to the Tour
          </AlertDialogTitle>
          <AlertDialogDescription className="text-muted-foreground mt-2 text-center text-sm">
            Take a quick tour to learn about the key features and functionality
            of this application.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="mt-6 space-y-3">
          <Button onClick={startTour} className="w-full">
            Start Tour
          </Button>
          <Button onClick={handleSkip} variant="ghost" className="w-full">
            Skip Tour
          </Button>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  );
}
