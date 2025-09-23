import { ChatContainer } from "./chat-container";
import { PromptSuggestions } from "./prompt-suggestions";
import { MessageInputWrapper } from "./message-input-wrapper";
import { ErrorDisplay } from "./error-display";
import { ChatStatus } from "./chat-status";
import { cn } from "@/lib/shared/utils";

interface WelcomeScreenProps {
  className?: string;
  suggestions: string[];
  onSuggestionClick: (text: string) => void;
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { files?: FileList },
  ) => void;
  input: string;
  handleInputChange: React.ChangeEventHandler<HTMLTextAreaElement>;
  isGenerating: boolean;
  isTyping: boolean;
  hasReachedLimit: boolean;
  transcribeAudio?: (blob: Blob) => Promise<string>;
  isError?: boolean;
  error?: Error;
  onRetry?: () => void;
  onStop: () => void;
  // Chat status props
  shouldShowCounter: boolean;
  shouldShowWarning: boolean;
  limitStatus: "normal" | "low" | "warning" | "reached";
  usedMessages: number;
  displayRemainingMessages?: number;
  userId?: string;
}

export function WelcomeScreen({
  className,
  suggestions,
  onSuggestionClick,
  handleSubmit,
  input,
  handleInputChange,
  isGenerating,
  isTyping,
  hasReachedLimit,
  transcribeAudio,
  isError,
  error,
  onRetry,
  onStop,
  shouldShowCounter,
  shouldShowWarning,
  limitStatus,
  usedMessages,
  displayRemainingMessages,
  userId,
}: WelcomeScreenProps) {
  return (
    <ChatContainer
      className={cn(
        "min-h-screen flex flex-col items-center justify-center bg-background p-6",
        className,
      )}
    >
      <div className="w-full max-w-3xl mx-auto flex flex-col items-center">
        <PromptSuggestions
          label="Try these prompts ✨"
          onSuggestionClick={onSuggestionClick}
          suggestions={suggestions}
          enableCategories={true}
        />

        {/* Input area with integrated functions */}
        <MessageInputWrapper
          className="w-full mb-6"
          isPending={isGenerating || isTyping}
          handleSubmit={handleSubmit}
          hasReachedLimit={hasReachedLimit}
          value={input}
          onChange={handleInputChange}
          stop={onStop}
          isGenerating={isGenerating}
          transcribeAudio={transcribeAudio}
        />

        {/* Error feedback */}
        <ErrorDisplay
          isError={isError}
          error={error}
          onRetry={onRetry}
          className="w-full max-w-3xl mx-auto mb-4"
        />

        <ChatStatus
          shouldShowCounter={shouldShowCounter}
          shouldShowWarning={shouldShowWarning}
          limitStatus={limitStatus}
          usedMessages={usedMessages}
          displayRemainingMessages={displayRemainingMessages}
          userId={userId}
        />
      </div>
    </ChatContainer>
  );
}