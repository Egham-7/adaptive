import { forwardRef, useCallback, useState } from "react";
import type { ChatFormProps } from "./chat-types";
import { createFileList } from "./chat-utils";

export const ChatForm = forwardRef<HTMLFormElement, ChatFormProps>(
  ({ children, handleSubmit, className, hasReachedLimit = false }, ref) => {
    const [files, setFiles] = useState<File[] | null>(null);

    const onSubmit = useCallback(
      (event: React.FormEvent) => {
        if (hasReachedLimit) {
          event.preventDefault();
          return;
        }

        if (!files) {
          handleSubmit(event);
          return;
        }

        const fileList = createFileList(files);
        handleSubmit(event, { files: fileList });
        setFiles(null);
      },
      [hasReachedLimit, files, handleSubmit],
    );

    return (
      <form ref={ref} onSubmit={onSubmit} className={className}>
        {children({ files, setFiles })}
      </form>
    );
  },
);
ChatForm.displayName = "ChatForm";