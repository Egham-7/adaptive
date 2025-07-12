import type { UIMessage } from "@ai-sdk/react";

/**
 * Message utility functions for chat operations
 */

/**
 * Compares two message arrays to determine if they represent the same messages
 */
export function areMessagesEqual<T>(current: UIMessage<T>[], incoming: UIMessage<T>[]): boolean {
  if (current.length !== incoming.length) return false;
  
  return current.every((msg, index) => {
    const incomingMsg = incoming[index];
    return msg.id === incomingMsg?.id && 
           msg.role === incomingMsg?.role &&
           JSON.stringify(msg.parts) === JSON.stringify(incomingMsg?.parts);
  });
}

/**
 * Gets the text content from a message
 */
export function getMessageText<T>(message: UIMessage<T>): string {
  const textPart = message.parts?.find(p => p.type === "text") as { text: string } | undefined;
  return textPart?.text || "";
}

/**
 * Checks if a message is the last message in an array
 */
export function isLastMessage<T>(message: UIMessage<T>, messages: UIMessage<T>[]): boolean {
  const lastMessage = messages[messages.length - 1];
  return lastMessage?.id === message.id;
}

/**
 * Checks if a message should show streaming animation
 */
export function shouldShowStreaming<T>(
  message: UIMessage<T>, 
  messages: UIMessage<T>[], 
  isGenerating: boolean
): boolean {
  return isGenerating && 
         message.role === "assistant" && 
         isLastMessage(message, messages);
}

/**
 * Gets user messages count from message array
 */
export function getUserMessageCount<T>(messages: UIMessage<T>[]): number {
  return messages.filter(m => m.role === "user").length;
}

/**
 * Finds message index by ID
 */
export function findMessageIndex<T>(messages: UIMessage<T>[], messageId: string): number {
  return messages.findIndex(m => m.id === messageId);
}

/**
 * Gets messages slice from index
 */
export function getMessagesFrom<T>(messages: UIMessage<T>[], fromIndex: number): UIMessage<T>[] {
  return fromIndex === -1 ? messages : messages.slice(0, fromIndex);
}

/**
 * Gets the last assistant message
 */
export function getLastAssistantMessage<T>(messages: UIMessage<T>[]): UIMessage<T> | undefined {
  return messages.findLast(m => m.role === "assistant");
}