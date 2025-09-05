import { useCallback } from "react";
import type { UIMessage } from "@ai-sdk/react";
import type { MessageCapabilities } from "../chat-types";
import { MESSAGE_ROLES } from "../constants/chat-constants";

interface MessageCapabilitiesHookProps {
  isGenerating: boolean;
  editingMessageId: string | null;
  canRate: boolean;
}

/**
 * Hook for determining message capabilities
 * Centralizes the logic for what actions are available for each message
 */
export function useMessageCapabilities({
  isGenerating,
  editingMessageId,
  canRate,
}: MessageCapabilitiesHookProps) {
  const getMessageCapabilities = useCallback((message: UIMessage): MessageCapabilities => {
    const isUserMessage = message.role === MESSAGE_ROLES.USER;
    const isEditing = editingMessageId === message.id;
    
    return {
      canEdit: isUserMessage && !isGenerating,
      canRetry: isUserMessage && !isGenerating,
      canDelete: !isGenerating,
      canRate: canRate && message.role === MESSAGE_ROLES.ASSISTANT,
      isEditing,
    };
  }, [isGenerating, editingMessageId, canRate]);

  return { getMessageCapabilities };
}