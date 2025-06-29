import type { UIMessage } from "@ai-sdk/react";

export function getMessageContent(message: UIMessage): string {
  const textPart = message.parts?.find((p) => p.type === "text") as
    | { text: string }
    | undefined;
  return textPart?.text || "";
}

export function createFileList(files: File[] | FileList): FileList {
  const dataTransfer = new DataTransfer();
  for (const file of Array.from(files)) {
    dataTransfer.items.add(file);
  }
  return dataTransfer.files;
}

export function getErrorMessage(error: Error): string {
  try {
    const parsed = JSON.parse(error.message);
    return parsed.error || parsed.message || error.message;
  } catch {
    return error.message || "An unexpected error occurred.";
  }
}