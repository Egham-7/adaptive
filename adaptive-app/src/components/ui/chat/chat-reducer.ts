import type { MessageAction, MessageState } from "./chat-types";

export function messageReducer(
  state: MessageState,
  action: MessageAction,
): MessageState {
  switch (action.type) {
    case "CANCEL_TOOL_INVOCATIONS": {
      const messageIndex = state.messages.findIndex(
        (m) => m.id === action.messageId,
      );
      if (messageIndex === -1) return state;

      const message = state.messages[messageIndex];
      if (!message?.parts?.length) return state;

      let needsUpdate = false;
      const updatedParts = message.parts.map((part) => {
        if (
          part.type === "tool-invocation" &&
          part.toolInvocation.state === "call"
        ) {
          needsUpdate = true;
          return {
            ...part,
            toolInvocation: {
              ...part.toolInvocation,
              state: "result" as const,
              result: "Tool execution was cancelled",
              isError: true,
            },
          };
        }
        return part;
      });

      if (!needsUpdate) return state;

      const newMessages = [...state.messages];
      newMessages[messageIndex] = { ...message, parts: updatedParts };

      return { ...state, messages: newMessages };
    }

    case "DELETE_MESSAGE_AND_AFTER": {
      const messageIndex = state.messages.findIndex(
        (m) => m.id === action.messageId,
      );
      const newMessages =
        messageIndex === -1
          ? state.messages
          : state.messages.slice(0, messageIndex);

      return {
        ...state,
        messages: newMessages,
        editingMessageId: null,
        editingContent: "",
      };
    }

    case "SET_MESSAGES":
      return { ...state, messages: action.messages };

    case "EDIT_MESSAGE":
      return {
        ...state,
        editingMessageId: action.messageId,
        editingContent: action.content,
      };

    case "UPDATE_EDITING_CONTENT":
      return {
        ...state,
        editingContent: action.content,
      };

    case "RETRY_MESSAGE": {
      const messageIndex = state.messages.findIndex(
        (m) => m.id === action.messageId,
      );
      const newMessages =
        messageIndex === -1
          ? state.messages
          : state.messages.slice(0, messageIndex);

      return {
        ...state,
        messages: newMessages,
        editingMessageId: null,
        editingContent: "",
      };
    }

    case "CLEAR_EDITING":
      return {
        ...state,
        editingMessageId: null,
        editingContent: "",
      };

    default:
      return state;
  }
}