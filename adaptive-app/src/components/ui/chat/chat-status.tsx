import { DAILY_MESSAGE_LIMIT } from "@/lib/chat/message-limits";
import SubscribeButton from "@/app/_components/stripe/subscribe-button";
import { cn } from "@/lib/utils";

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
      <span
        className={cn(
          "text-xs px-2 py-1 rounded-full",
          limitStatus === "reached"
            ? "bg-destructive/10 text-destructive"
            : limitStatus === "warning"
              ? "bg-secondary/10 text-secondary-foreground"
              : "bg-muted text-muted-foreground",
        )}
      >
        {usedMessages}/{DAILY_MESSAGE_LIMIT} messages used today
      </span>
    </div>
  ) : null;

  const MessageLimitWarning = shouldShowWarning ? (
    <div className="mx-4 mt-8 text-center">
      <div className="inline-block rounded-lg border border-secondary/20 bg-secondary/10 p-4 text-left">
        <p className="text-sm text-secondary-foreground">
          {displayRemainingMessages !== undefined && displayRemainingMessages > 0
            ? `You have ${displayRemainingMessages} messages remaining today.`
            : "You've reached your daily message limit."}
          {userId && (
            <SubscribeButton
              variant="link"
              className="ml-1 text-secondary-foreground hover:text-secondary-foreground/80"
            >
              Upgrade to Pro
            </SubscribeButton>
          )}{" "}
          for unlimited messages.
        </p>
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