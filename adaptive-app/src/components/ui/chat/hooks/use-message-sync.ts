import { useEffect, useRef } from "react";
import type { UIMessage } from "@ai-sdk/react";
import { areMessagesEqual } from "../utils/message-utils";

/**
 * Hook to synchronize external messages with internal state
 * Uses efficient comparison to prevent unnecessary updates
 */
export function useMessageSync(
  externalMessages: UIMessage[],
  internalMessages: UIMessage[],
  onSync: (messages: UIMessage[]) => void
) {
  const lastSyncedRef = useRef<UIMessage[]>(externalMessages);

  useEffect(() => {
    // Only sync if messages have meaningfully changed
    const hasChanged = !areMessagesEqual(internalMessages, externalMessages);
    const isNewReference = lastSyncedRef.current !== externalMessages;
    
    if (hasChanged && isNewReference) {
      onSync(externalMessages);
      lastSyncedRef.current = externalMessages;
    }
  }, [externalMessages, internalMessages]);
}