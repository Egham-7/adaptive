import { forwardRef, useCallback, useState } from "react";
import type { ChatFormProps } from "./chat-types";
import { createFileList } from "./chat-utils";

export const ChatForm = forwardRef<HTMLFormElement, ChatFormProps>(
  ({ children, handleSubmit, className, hasReachedLimit = false }, ref) => {
    const [files, setFiles] = useState<File[] | null>(null);
    const [searchEnabled, setSearchEnabled] = useState(false);

    const onSubmit = useCallback(
      (event: React.FormEvent) => {
        if (hasReachedLimit) {
          event.preventDefault();
          return;
        }

        if (!files) {
          handleSubmit(event, { searchEnabled });
          return;
        }

        const fileList = createFileList(files);
        handleSubmit(event, { files: fileList, searchEnabled });
        setFiles(null);
      },
      [hasReachedLimit, files, handleSubmit, searchEnabled],
    );

    return (
      <form ref={ref} onSubmit={onSubmit} className={className}>
        {children({ files, setFiles, searchEnabled, setSearchEnabled })}
      </form>
    );
  },
);
ChatForm.displayName = "ChatForm";