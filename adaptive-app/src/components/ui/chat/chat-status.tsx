import { DAILY_MESSAGE_LIMIT } from "@/lib/chat/message-limits";
import SubscribeButton from "@/app/_components/stripe/subscribe-button";
import { cn } from "@/lib/shared/utils";
import { BsChatSquare, BsExclamationTriangle } from "react-icons/bs";
import { FaCrown } from "react-icons/fa";

interface ChatStatusProps {
  shouldShowCounter: boolean;
  shouldShowWarning: boolean;
  limitStatus: "normal" | "low" | "warning" | "reached";
  usedMessages: number;
  displayRemainingMessages?: number;
  userId?: string;
}

export function ChatStatus({
  shouldShowCounter,
  shouldShowWarning,
  limitStatus,
  usedMessages,
  displayRemainingMessages,
  userId,
}: ChatStatusProps) {
  const MessageCounter = shouldShowCounter ? (
    <div className="mx-4 mb-2 text-center">
      <div className="inline-flex items-center gap-1.5">
        <BsChatSquare className="h-3 w-3" />
        <span
          className={cn(
            "text-xs px-2 py-1 rounded-full font-medium",
            limitStatus === "reached"
              ? "bg-destructive/10 text-destructive"
              : limitStatus === "warning"
                ? "bg-orange-100 text-orange-700 dark:bg-orange-900/20 dark:text-orange-400"
                : "bg-muted text-muted-foreground",
          )}
        >
          {usedMessages}/{DAILY_MESSAGE_LIMIT} messages used today
        </span>
      </div>
    </div>
  ) : null;

  const MessageLimitWarning = shouldShowWarning ? (
    <div className="mx-4 mt-6 text-center">
      <div className={cn(
        "inline-flex items-start gap-3 rounded-lg border p-4 text-left max-w-md",
        limitStatus === "reached"
          ? "border-destructive/20 bg-destructive/5"
          : "border-orange-200 bg-orange-50 dark:border-orange-900/20 dark:bg-orange-900/10"
      )}>
        <div className="flex-shrink-0 mt-0.5">
          {limitStatus === "reached" ? (
            <BsExclamationTriangle className={cn(
              "h-4 w-4",
              "text-destructive"
            )} />
          ) : (
            <BsExclamationTriangle className="h-4 w-4 text-orange-600 dark:text-orange-400" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <p className={cn(
            "text-sm font-medium mb-1",
            limitStatus === "reached"
              ? "text-destructive"
              : "text-orange-700 dark:text-orange-400"
          )}>
            {displayRemainingMessages !== undefined && displayRemainingMessages > 0
              ? `${displayRemainingMessages} messages remaining today`
              : "Daily message limit reached"}
          </p>
          <p className={cn(
            "text-xs",
            limitStatus === "reached"
              ? "text-destructive/80"
              : "text-orange-600 dark:text-orange-400/80"
          )}>
            {userId && (
              <>
                <SubscribeButton
                  variant="link"
                  className={cn(
                    "p-0 h-auto font-medium underline",
                    limitStatus === "reached"
                      ? "text-destructive hover:text-destructive/80"
                      : "text-orange-700 hover:text-orange-600 dark:text-orange-400 dark:hover:text-orange-300"
                  )}
                >
                  <FaCrown className="h-3 w-3 mr-1 inline" />
                  Upgrade to Pro
                </SubscribeButton>
                {" "}for unlimited messages.
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  ) : null;

  return (
    <>
      {MessageCounter}
      {MessageLimitWarning}
    </>
  );
}