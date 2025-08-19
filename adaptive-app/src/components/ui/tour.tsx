import { AnimatePresence, motion } from "motion/react";
import type React from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
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
import { useTourCompletion } from "@/hooks/use-tour-completion";

import { Torus } from "lucide-react";

// Types and interfaces
export interface TourStep {
  content: React.ReactNode;
  selectorId: string;
  width?: number;
  height?: number;
  onClickWithinArea?: () => void;
  position?: "top" | "bottom" | "left" | "right" | "auto";
}

interface TourContextType {
  currentStep: number;
  totalSteps: number;
  nextStep: () => void;
  previousStep: () => void;
  endTour: () => void;
  isActive: boolean;
  startTour: () => void;
  setSteps: (steps: TourStep[]) => void;
  steps: TourStep[];
  isTourCompleted: boolean;
  setIsTourCompleted: (completed: boolean) => void;
}

interface TourProviderProps {
  children: React.ReactNode;
  onComplete?: () => void;
  className?: string;
}

interface ElementRect {
  top: number;
  left: number;
  width: number;
  height: number;
  right: number;
  bottom: number;
}

interface ViewportInfo {
  width: number;
  height: number;
  scrollX: number;
  scrollY: number;
  top: number;
  left: number;
  right: number;
  bottom: number;
}

interface ContentPosition {
  top: number;
  left: number;
  width: number;
  height: number;
  actualPosition: "top" | "bottom" | "left" | "right";
}

// Constants
const CONTENT_PADDING = 16;
const VIEWPORT_MARGIN = 24; // Minimum distance from viewport edges
const DEFAULT_CONTENT_WIDTH = 320;
const DEFAULT_CONTENT_HEIGHT = 180;
const ARROW_SIZE = 8;

const TourContext = createContext<TourContextType | null>(null);

// Utility functions
function getViewportInfo(): ViewportInfo {
  const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
  const scrollY = window.pageYOffset || document.documentElement.scrollTop;
  const width = window.innerWidth;
  const height = window.innerHeight;
  
  return {
    width,
    height,
    scrollX,
    scrollY,
    top: scrollY,
    left: scrollX,
    right: scrollX + width,
    bottom: scrollY + height,
  };
}

function getElementRect(selectorId: string): ElementRect | null {
  const element = document.getElementById(selectorId);
  if (!element) return null;
  
  const rect = element.getBoundingClientRect();
  const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
  const scrollY = window.pageYOffset || document.documentElement.scrollTop;
  
  return {
    top: rect.top + scrollY,
    left: rect.left + scrollX,
    width: rect.width,
    height: rect.height,
    right: rect.left + scrollX + rect.width,
    bottom: rect.top + scrollY + rect.height,
  };
}

function calculateOptimalPosition(
  elementRect: ElementRect,
  preferredPosition: "top" | "bottom" | "left" | "right" | "auto",
  viewport: ViewportInfo,
  contentWidth: number = DEFAULT_CONTENT_WIDTH,
  contentHeight: number = DEFAULT_CONTENT_HEIGHT,
): ContentPosition {
  
  // Calculate all possible positions
  const positions = {
    top: {
      top: elementRect.top - contentHeight - CONTENT_PADDING,
      left: elementRect.left + elementRect.width / 2 - contentWidth / 2,
      width: contentWidth,
      height: contentHeight,
      actualPosition: "top" as const,
    },
    bottom: {
      top: elementRect.bottom + CONTENT_PADDING,
      left: elementRect.left + elementRect.width / 2 - contentWidth / 2,
      width: contentWidth,
      height: contentHeight,
      actualPosition: "bottom" as const,
    },
    left: {
      top: elementRect.top + elementRect.height / 2 - contentHeight / 2,
      left: elementRect.left - contentWidth - CONTENT_PADDING,
      width: contentWidth,
      height: contentHeight,
      actualPosition: "left" as const,
    },
    right: {
      top: elementRect.top + elementRect.height / 2 - contentHeight / 2,
      left: elementRect.right + CONTENT_PADDING,
      width: contentWidth,
      height: contentHeight,
      actualPosition: "right" as const,
    },
  };

  // Check if a position fits within viewport bounds
  function positionFitsInViewport(pos: ContentPosition): boolean {
    const fitsHorizontally = 
      pos.left >= viewport.left + VIEWPORT_MARGIN && 
      pos.left + pos.width <= viewport.right - VIEWPORT_MARGIN;
    
    const fitsVertically = 
      pos.top >= viewport.top + VIEWPORT_MARGIN && 
      pos.top + pos.height <= viewport.bottom - VIEWPORT_MARGIN;
    
    return fitsHorizontally && fitsVertically;
  }

  // Priority order for fallback positions
  const getPriorityOrder = (preferred: typeof preferredPosition): (keyof typeof positions)[] => {
    switch (preferred) {
      case "top": return ["top", "bottom", "right", "left"];
      case "bottom": return ["bottom", "top", "right", "left"];
      case "left": return ["left", "right", "bottom", "top"];
      case "right": return ["right", "left", "bottom", "top"];
      case "auto": return ["bottom", "top", "right", "left"]; // Default to bottom for auto
      default: return ["bottom", "top", "right", "left"];
    }
  };

  // Try positions in priority order
  const priorityOrder = getPriorityOrder(preferredPosition);
  
  for (const positionKey of priorityOrder) {
    const position = positions[positionKey];
    if (positionFitsInViewport(position)) {
      return position;
    }
  }

  // If no position fits perfectly, constrain the preferred position to viewport
  const fallbackPosition = preferredPosition === "auto" ? "bottom" : preferredPosition;
  const constrainedPosition = { ...positions[fallbackPosition] };

  // Constrain horizontally
  if (constrainedPosition.left < viewport.left + VIEWPORT_MARGIN) {
    constrainedPosition.left = viewport.left + VIEWPORT_MARGIN;
  } else if (constrainedPosition.left + constrainedPosition.width > viewport.right - VIEWPORT_MARGIN) {
    constrainedPosition.left = viewport.right - VIEWPORT_MARGIN - constrainedPosition.width;
  }

  // Constrain vertically
  if (constrainedPosition.top < viewport.top + VIEWPORT_MARGIN) {
    constrainedPosition.top = viewport.top + VIEWPORT_MARGIN;
  } else if (constrainedPosition.top + constrainedPosition.height > viewport.bottom - VIEWPORT_MARGIN) {
    constrainedPosition.top = viewport.bottom - VIEWPORT_MARGIN - constrainedPosition.height;
  }

  return constrainedPosition;
}

// Custom hook for tour positioning
function useTourPositioning(
  currentStep: number,
  steps: TourStep[],
  isActive: boolean
) {
  const [elementRect, setElementRect] = useState<ElementRect | null>(null);
  const [contentPosition, setContentPosition] = useState<ContentPosition | null>(null);
  const positionUpdateTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);

  const updatePositions = useCallback(() => {
    if (!isActive || currentStep < 0 || currentStep >= steps.length) {
      setElementRect(null);
      setContentPosition(null);
      return;
    }

    const step = steps[currentStep];
    if (!step) return;

    // Clear any pending updates
    if (positionUpdateTimeoutRef.current) {
      clearTimeout(positionUpdateTimeoutRef.current);
    }

    // Small delay to ensure DOM is ready
    positionUpdateTimeoutRef.current = setTimeout(() => {
      const rect = getElementRect(step.selectorId);
      if (!rect) return;

      const viewport = getViewportInfo();
      const contentPos = calculateOptimalPosition(
        rect,
        step.position || "auto",
        viewport,
        step.width || DEFAULT_CONTENT_WIDTH,
        step.height || DEFAULT_CONTENT_HEIGHT
      );

      setElementRect(rect);
      setContentPosition(contentPos);

      // Smooth scroll to element
      const element = document.getElementById(step.selectorId);
      if (element) {
        element.scrollIntoView({
          behavior: "smooth",
          block: "center",
          inline: "nearest",
        });
      }
    }, 100);
  }, [currentStep, steps, isActive]);

  // Update positions when dependencies change
  useEffect(() => {
    updatePositions();
    
    return () => {
      if (positionUpdateTimeoutRef.current) {
        clearTimeout(positionUpdateTimeoutRef.current);
      }
    };
  }, [updatePositions]);

  // Handle window resize and scroll
  useEffect(() => {
    if (!isActive) return;

    const handleResize = () => updatePositions();
    const handleScroll = () => updatePositions();

    window.addEventListener("resize", handleResize);
    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("scroll", handleScroll);
    };
  }, [isActive, updatePositions]);

  return { elementRect, contentPosition };
}

export function TourProvider({
  children,
  onComplete,
  className,
}: TourProviderProps) {
  const { isTourCompleted, setIsTourCompleted, isLoading } = useTourCompletion();
  const [steps, setSteps] = useState<TourStep[]>([]);
  const [currentStep, setCurrentStep] = useState(-1);
  
  const isActive = currentStep >= 0 && currentStep < steps.length;
  const { elementRect, contentPosition } = useTourPositioning(currentStep, steps, isActive);

  // Memoized callbacks for better performance
  const nextStep = useCallback(async () => {
    if (currentStep >= steps.length - 1) {
      await setIsTourCompleted(true);
      onComplete?.();
      setCurrentStep(-1);
    } else {
      setCurrentStep((prev) => prev + 1);
    }
  }, [steps.length, onComplete, currentStep, setIsTourCompleted]);

  const previousStep = useCallback(() => {
    setCurrentStep((prev) => Math.max(0, prev - 1));
  }, []);

  const endTour = useCallback(() => {
    setCurrentStep(-1);
  }, []);

  const startTour = useCallback(() => {
    if (isTourCompleted || steps.length === 0) return;
    setCurrentStep(0);
  }, [isTourCompleted, steps.length]);

  const handleStepsChange = useCallback((newSteps: TourStep[]) => {
    setSteps(newSteps);
  }, []);

  const handleSetIsTourCompleted = useCallback(
    async (completed: boolean) => {
      await setIsTourCompleted(completed);
    },
    [setIsTourCompleted],
  );

  // Handle click within highlighted area
  const handleAreaClick = useCallback(
    (e: MouseEvent) => {
      if (!isActive || !elementRect || currentStep < 0) return;
      
      const step = steps[currentStep];
      if (!step?.onClickWithinArea) return;

      const clickX = e.clientX + window.pageXOffset;
      const clickY = e.clientY + window.pageYOffset;
      
      const isWithinBounds =
        clickX >= elementRect.left &&
        clickX <= elementRect.right &&
        clickY >= elementRect.top &&
        clickY <= elementRect.bottom;

      if (isWithinBounds) {
        step.onClickWithinArea();
      }
    },
    [isActive, elementRect, currentStep, steps],
  );

  // Add click event listener
  useEffect(() => {
    if (!isActive) return;
    
    window.addEventListener("click", handleAreaClick);
    return () => window.removeEventListener("click", handleAreaClick);
  }, [isActive, handleAreaClick]);

  // Context value with memoization
  const contextValue = useMemo(
    () => ({
      currentStep,
      totalSteps: steps.length,
      nextStep,
      previousStep,
      endTour,
      isActive,
      startTour,
      setSteps: handleStepsChange,
      steps,
      isTourCompleted,
      setIsTourCompleted: handleSetIsTourCompleted,
    }),
    [
      currentStep,
      steps.length,
      nextStep,
      previousStep,
      endTour,
      isActive,
      startTour,
      handleStepsChange,
      steps,
      isTourCompleted,
      handleSetIsTourCompleted,
    ],
  );

  if (isLoading) return null;

  return (
    <TourContext.Provider value={contextValue}>
      {children}
      <TourOverlay
        isActive={isActive}
        elementRect={elementRect}
        contentPosition={contentPosition}
        currentStep={currentStep}
        steps={steps}
        className={className}
        onNext={nextStep}
        onPrevious={previousStep}
      />
    </TourContext.Provider>
  );
}

// Separate component for tour overlay to improve performance
interface TourOverlayProps {
  isActive: boolean;
  elementRect: ElementRect | null;
  contentPosition: ContentPosition | null;
  currentStep: number;
  steps: TourStep[];
  className?: string;
  onNext: () => void;
  onPrevious: () => void;
}

function TourOverlay({
  isActive,
  elementRect,
  contentPosition,
  currentStep,
  steps,
  className,
  onNext,
  onPrevious,
}: TourOverlayProps) {
  if (!isActive || !elementRect || !contentPosition || currentStep < 0) {
    return null;
  }

  const currentStepData = steps[currentStep];
  if (!currentStepData) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 pointer-events-none"
        style={{
          background: `radial-gradient(circle at ${elementRect.left + elementRect.width / 2}px ${elementRect.top + elementRect.height / 2}px, transparent ${Math.max(elementRect.width, elementRect.height) / 2 + 20}px, rgba(0, 0, 0, 0.5) ${Math.max(elementRect.width, elementRect.height) / 2 + 40}px)`,
        }}
      >
        {/* Highlight border around target element */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className={cn("absolute border-2 border-primary rounded-lg", className)}
          style={{
            top: elementRect.top - 2,
            left: elementRect.left - 2,
            width: elementRect.width + 4,
            height: elementRect.height + 4,
          }}
        />

        {/* Tour content */}
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.95 }}
          animate={{
            opacity: 1,
            y: 0,
            scale: 1,
            top: contentPosition.top,
            left: contentPosition.left,
          }}
          exit={{ opacity: 0, y: 10, scale: 0.95 }}
          transition={{
            duration: 0.2,
            ease: "easeOut",
          }}
          className="absolute bg-background border rounded-lg shadow-lg pointer-events-auto"
          style={{
            width: contentPosition.width,
            minHeight: 120,
            maxWidth: "calc(100vw - 48px)",
          }}
        >
          {/* Content container */}
          <div className="p-4">
            {/* Step counter */}
            <div className="absolute top-3 right-4 text-xs text-muted-foreground">
              {currentStep + 1} / {steps.length}
            </div>

            {/* Step content */}
            <AnimatePresence mode="wait">
              <motion.div
                key={`tour-content-${currentStep}`}
                initial={{ opacity: 0, filter: "blur(4px)" }}
                animate={{ opacity: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, filter: "blur(4px)" }}
                transition={{ duration: 0.15 }}
                className="pr-12 pb-4"
              >
                {currentStepData.content}
              </motion.div>
            </AnimatePresence>

            {/* Navigation buttons */}
            <div className="flex justify-between items-center mt-4">
              <div>
                {currentStep > 0 && (
                  <Button
                    onClick={onPrevious}
                    variant="ghost"
                    size="sm"
                    className="text-sm"
                  >
                    Previous
                  </Button>
                )}
              </div>
              <Button
                onClick={onNext}
                variant="default"
                size="sm"
                className="text-sm"
              >
                {currentStep === steps.length - 1 ? "Finish" : "Next"}
              </Button>
            </div>
          </div>

          {/* Arrow indicator pointing to target */}
          <TourArrow
            elementRect={elementRect}
            contentPosition={contentPosition}
            position={contentPosition.actualPosition}
          />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

// Arrow component for better visual connection
interface TourArrowProps {
  elementRect: ElementRect;
  contentPosition: ContentPosition;
  position: "top" | "bottom" | "left" | "right";
}

function TourArrow({ elementRect, contentPosition, position }: TourArrowProps) {
  const arrowStyle: React.CSSProperties = {};
  const arrowClass = "absolute w-0 h-0 border-solid";
  
  switch (position) {
    case "top":
      arrowStyle.bottom = -ARROW_SIZE;
      arrowStyle.left = "50%";
      arrowStyle.transform = "translateX(-50%)";
      arrowStyle.borderLeft = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderRight = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderTop = `${ARROW_SIZE}px solid hsl(var(--border))`;
      break;
    case "bottom":
      arrowStyle.top = -ARROW_SIZE;
      arrowStyle.left = "50%";
      arrowStyle.transform = "translateX(-50%)";
      arrowStyle.borderLeft = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderRight = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderBottom = `${ARROW_SIZE}px solid hsl(var(--border))`;
      break;
    case "left":
      arrowStyle.right = -ARROW_SIZE;
      arrowStyle.top = "50%";
      arrowStyle.transform = "translateY(-50%)";
      arrowStyle.borderTop = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderBottom = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderLeft = `${ARROW_SIZE}px solid hsl(var(--border))`;
      break;
    case "right":
      arrowStyle.left = -ARROW_SIZE;
      arrowStyle.top = "50%";
      arrowStyle.transform = "translateY(-50%)";
      arrowStyle.borderTop = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderBottom = `${ARROW_SIZE}px solid transparent`;
      arrowStyle.borderRight = `${ARROW_SIZE}px solid hsl(var(--border))`;
      break;
  }

  return <div className={arrowClass} style={arrowStyle} />;
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
  const { startTour, steps, isTourCompleted, currentStep, setIsTourCompleted } =
    useTour();

  if (isTourCompleted || steps.length === 0 || currentStep > -1) {
    return null;
  }

  const handleSkip = async () => {
    setIsOpen(false);
    await setIsTourCompleted(true);
  };

  const handleStartTour = () => {
    startTour();
    setIsOpen(false);
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
            Welcome to Adaptive
          </AlertDialogTitle>
          <AlertDialogDescription className="text-muted-foreground mt-2 text-center text-sm">
            Take a guided tour to learn about the navigation, dashboard metrics,
            and key features of the API platform.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="mt-6 space-y-3">
          <Button onClick={handleStartTour} className="w-full">
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
