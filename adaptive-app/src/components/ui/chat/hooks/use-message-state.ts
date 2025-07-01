import { useCallback, useReducer, useMemo } from "react";
import type { UIMessage } from "@ai-sdk/react";
import { messageReducer } from "../chat-reducer";
import type { MessageState, MessageComputedState } from "../chat-types";
import { getLastAssistantMessage, getUserMessageCount } from "../utils/message-utils";

/**
 * Hook for managing message state with computed values
 * Provides optimized selectors to prevent unnecessary re-renders
 */
export function useMessageState(initialMessages: UIMessage[]) {
  const [state, dispatch] = useReducer(messageReducer, {
    messages: initialMessages,
    editingMessageId: null,
    editingContent: "",
  });

  // Memoized computed state to prevent unnecessary recalculations
  const computedState = useMemo((): MessageComputedState => ({
    lastMessage: state.messages[state.messages.length - 1],
    lastAssistantMessage: getLastAssistantMessage(state.messages),
    isEmpty: state.messages.length === 0,
    userMessageCount: getUserMessageCount(state.messages),
  }), [state.messages]);

  // Action creators
  const actions = useMemo(() => ({
    setMessages: (messages: UIMessage[]) => 
      dispatch({ type: "SET_MESSAGES", messages }),
    
    startEditing: (messageId: string, content: string) => 
      dispatch({ type: "EDIT_MESSAGE", messageId, content }),
    
    updateEditingContent: (messageId: string, content: string) => 
      dispatch({ type: "UPDATE_EDITING_CONTENT", content }),
    
    clearEditing: () => 
      dispatch({ type: "CLEAR_EDITING" }),
    
    cancelToolInvocations: (messageId: string) => 
      dispatch({ type: "CANCEL_TOOL_INVOCATIONS", messageId }),
    
    deleteMessageAndAfter: (messageId: string) => 
      dispatch({ type: "DELETE_MESSAGE_AND_AFTER", messageId }),
    
    retryMessage: (messageId: string) => 
      dispatch({ type: "RETRY_MESSAGE", messageId }),
  }), []);

  return {
    state,
    computed: computedState,
    actions,
  };
}