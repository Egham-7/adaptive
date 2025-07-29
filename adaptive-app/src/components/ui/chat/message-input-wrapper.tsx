import { ChatForm } from "./chat-form";
import { MessageInput } from "./message-input";

interface MessageInputWrapperProps {
  className?: string;
  isPending: boolean;
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { files?: FileList },
  ) => void;
  hasReachedLimit?: boolean;
  value: string;
  onChange: React.ChangeEventHandler<HTMLTextAreaElement>;
  stop?: () => void;
  isGenerating: boolean;
  transcribeAudio?: (blob: Blob) => Promise<string>;
}

export function MessageInputWrapper({
  className,
  isPending,
  handleSubmit,
  hasReachedLimit = false,
  value,
  onChange,
  stop,
  isGenerating,
  transcribeAudio,
}: MessageInputWrapperProps) {
  const handleStop = () => {
    stop?.();
  };

  return (
    <div className={`w-full ${className}`}>
      <ChatForm
        isPending={isPending}
        handleSubmit={handleSubmit}
        hasReachedLimit={hasReachedLimit}
      >
        {({ files, setFiles }) => (
          <MessageInput
            value={value}
            onChange={onChange}
            allowAttachments
            files={files}
            setFiles={setFiles}
            stop={handleStop}
            isGenerating={isGenerating}
            transcribeAudio={transcribeAudio}
            disabled={hasReachedLimit}
            enableAdvancedFeatures={true}
            placeholder={
              hasReachedLimit
                ? "Daily message limit reached - upgrade to continue"
                : "Ask me anything..."
            }
          />
        )}
      </ChatForm>
    </div>
  );
}