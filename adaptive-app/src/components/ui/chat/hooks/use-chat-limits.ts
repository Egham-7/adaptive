import { useMemo } from "react";
import type { UIMessage } from "@ai-sdk/react";
import type { ChatLimitsState, ChatLimitsProps } from "../chat-types";
import { DAILY_MESSAGE_LIMIT } from "@/lib/chat/message-limits";
import { CHAT_LIMITS, CHAT_STATUS } from "../constants/chat-constants";
import { useOptimisticMessageCount } from "./use-optimistic-message-count";

/**
 * Hook for managing chat limits and usage tracking
 * Provides optimized state for message limits with proper typing
 */
export function useChatLimits(
  messages: UIMessage[],
  props: ChatLimitsProps
): ChatLimitsState {
  const {
    hasReachedLimit = false,
    remainingMessages,
    isUnlimited = true,
    limitsLoading = false,
  } = props;

  const optimisticRemainingMessages = useOptimisticMessageCount(
    messages,
    remainingMessages
  );

  const displayRemainingMessages = optimisticRemainingMessages ?? remainingMessages;
  
  return useMemo((): ChatLimitsState => {
    const usedMessages = displayRemainingMessages !== undefined
      ? DAILY_MESSAGE_LIMIT - displayRemainingMessages
      : 0;

    const shouldShowCounter = !limitsLoading && 
                             !isUnlimited && 
                             displayRemainingMessages !== undefined;

    const shouldShowWarning = shouldShowCounter &&
                             displayRemainingMessages !== undefined &&
                             displayRemainingMessages <= CHAT_LIMITS.LOW_REMAINING_THRESHOLD;

    const limitStatus = (() => {
      if (hasReachedLimit) return CHAT_STATUS.REACHED;
      if (displayRemainingMessages !== undefined) {
        if (displayRemainingMessages <= CHAT_LIMITS.WARNING_REMAINING_THRESHOLD) {
          return CHAT_STATUS.WARNING;
        }
        if (displayRemainingMessages <= CHAT_LIMITS.LOW_REMAINING_THRESHOLD) {
          return CHAT_STATUS.LOW;
        }
      }
      return CHAT_STATUS.NORMAL;
    })();

    return {
      hasReachedLimit,
      remainingMessages,
      isUnlimited,
      limitsLoading,
      usedMessages,
      displayRemainingMessages,
      shouldShowCounter,
      shouldShowWarning,
      limitStatus,
    };
  }, [
    hasReachedLimit,
    remainingMessages,
    isUnlimited,
    limitsLoading,
    displayRemainingMessages,
  ]);
}