import type { UIMessage } from "@ai-sdk/react";

/**
 * Message utility functions for chat operations
 */

/**
 * Compares two message arrays to determine if they represent the same messages
 */
export function areMessagesEqual(current: UIMessage[], incoming: UIMessage[]): boolean {
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
export function getMessageText(message: UIMessage): string {
  const textPart = message.parts?.find(p => p.type === "text") as { text: string } | undefined;
  return textPart?.text || "";
}

/**
 * Checks if a message is the last message in an array
 */
export function isLastMessage(message: UIMessage, messages: UIMessage[]): boolean {
  const lastMessage = messages[messages.length - 1];
  return lastMessage?.id === message.id;
}

/**
 * Checks if a message should show streaming animation
 */
export function shouldShowStreaming(
  message: UIMessage, 
  messages: UIMessage[], 
  isGenerating: boolean
): boolean {
  return isGenerating && 
         message.role === "assistant" && 
         isLastMessage(message, messages);
}

/**
 * Gets user messages count from message array
 */
export function getUserMessageCount(messages: UIMessage[]): number {
  return messages.filter(m => m.role === "user").length;
}

/**
 * Finds message index by ID
 */
export function findMessageIndex(messages: UIMessage[], messageId: string): number {
  return messages.findIndex(m => m.id === messageId);
}

/**
 * Gets messages slice from index
 */
export function getMessagesFrom(messages: UIMessage[], fromIndex: number): UIMessage[] {
  return fromIndex === -1 ? messages : messages.slice(0, fromIndex);
}

/**
 * Gets the last assistant message
 */
export function getLastAssistantMessage(messages: UIMessage[]): UIMessage | undefined {
  return messages.findLast(m => m.role === "assistant");
}