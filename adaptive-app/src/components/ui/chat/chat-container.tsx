import { forwardRef } from "react";
import { cn } from "@/lib/shared/utils";

export const ChatContainer = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("flex flex-col w-full", className)}
      {...props}
    />
  );
});
ChatContainer.displayName = "ChatContainer";