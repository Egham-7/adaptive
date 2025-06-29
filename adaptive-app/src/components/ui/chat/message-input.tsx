"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Info, Mic, Paperclip, Square, Search, BrainCircuit, Plus, Telescope, Zap } from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { omit } from "remeda";

import { AudioVisualizer } from "./audio-visualizer";
import { Button } from "@/components/ui/button";
import { FilePreview } from "./file-preview";
import { InterruptPrompt } from "./interrupt-prompt";
import { TextShimmerLoader } from "./loader";
import { useAudioRecording } from "@/hooks/use-audio-recording";
import { useAutosizeTextArea } from "@/hooks/use-autosize-textarea";
import { cn } from "@/lib/utils";

interface MessageInputBaseProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  value: string;
  submitOnEnter?: boolean;
  stop?: () => void;
  isGenerating: boolean;
  enableInterrupt?: boolean;
  transcribeAudio?: (blob: Blob) => Promise<string>;
  enableAdvancedFeatures?: boolean;
}

interface MessageInputWithoutAttachmentProps extends MessageInputBaseProps {
  allowAttachments?: false;
}

interface MessageInputWithAttachmentsProps extends MessageInputBaseProps {
  allowAttachments: true;
  files: File[] | null;
  setFiles: React.Dispatch<React.SetStateAction<File[] | null>>;
}

type MessageInputProps =
  | MessageInputWithoutAttachmentProps
  | MessageInputWithAttachmentsProps;

export function MessageInput({
  placeholder = "Ask AI...",
  className,
  onKeyDown: onKeyDownProp,
  submitOnEnter = true,
  stop,
  isGenerating,
  enableInterrupt = true,
  transcribeAudio,
  enableAdvancedFeatures = false,
  ...props
}: MessageInputProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [showInterruptPrompt, setShowInterruptPrompt] = useState(false);
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [deepResearchEnabled, setDeepResearchEnabled] = useState(false);
  const [reasonEnabled, setReasonEnabled] = useState(false);

  const {
    isListening,
    isSpeechSupported,
    isRecording,
    isTranscribing,
    audioStream,
    toggleListening,
    stopRecording,
  } = useAudioRecording({
    transcribeAudio,
    onTranscriptionComplete: (text) => {
      props.onChange?.({
        target: { value: text },
      } as React.ChangeEvent<HTMLTextAreaElement>);
    },
  });

  useEffect(() => {
    if (!isGenerating) {
      setShowInterruptPrompt(false);
    }
  }, [isGenerating]);

  const addFiles = (files: File[] | null) => {
    if (props.allowAttachments) {
      props.setFiles((currentFiles) => {
        if (currentFiles === null) {
          return files;
        }

        if (files === null) {
          return currentFiles;
        }

        return [...currentFiles, ...files];
      });
    }
  };

  const onDragOver = (event: React.DragEvent) => {
    if (props.allowAttachments !== true) return;
    event.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (event: React.DragEvent) => {
    if (props.allowAttachments !== true) return;
    event.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (event: React.DragEvent) => {
    setIsDragging(false);
    if (props.allowAttachments !== true) return;
    event.preventDefault();
    const dataTransfer = event.dataTransfer;
    if (dataTransfer.files.length) {
      addFiles(Array.from(dataTransfer.files));
    }
  };

  const onPaste = (event: React.ClipboardEvent) => {
    const items = event.clipboardData?.items;
    if (!items) return;

    const text = event.clipboardData.getData("text");
    if (text && text.length > 500 && props.allowAttachments) {
      event.preventDefault();
      const blob = new Blob([text], { type: "text/plain" });
      const file = new File([blob], "Pasted text", {
        type: "text/plain",
        lastModified: Date.now(),
      });
      addFiles([file]);
      return;
    }

    const files = Array.from(items)
      .map((item) => item.getAsFile())
      .filter((file) => file !== null);

    if (props.allowAttachments && files.length > 0) {
      addFiles(files);
    }
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (submitOnEnter && event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();

      if (isGenerating && stop && enableInterrupt) {
        if (showInterruptPrompt) {
          stop();
          setShowInterruptPrompt(false);
          event.currentTarget.form?.requestSubmit();
        } else if (
          props.value ||
          (props.allowAttachments && props.files?.length)
        ) {
          setShowInterruptPrompt(true);
          return;
        }
      }

      event.currentTarget.form?.requestSubmit();
    }

    onKeyDownProp?.(event);
  };

  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const [textAreaHeight, setTextAreaHeight] = useState<number>(0);

  useEffect(() => {
    if (textAreaRef.current) {
      setTextAreaHeight(textAreaRef.current.offsetHeight);
    }
  }, []);

  const showFileList =
    props.allowAttachments && props.files && props.files.length > 0;

  useAutosizeTextArea({
    ref: textAreaRef as React.RefObject<HTMLTextAreaElement>,
    maxHeight: 180,
    borderWidth: 1,
    dependencies: [props.value, showFileList],
  });

  return (
    <div
      className="relative flex w-full"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {enableInterrupt && (
        <InterruptPrompt
          isOpen={showInterruptPrompt}
          close={() => setShowInterruptPrompt(false)}
        />
      )}

      <RecordingPrompt
        isVisible={isRecording}
        onStopRecording={stopRecording}
      />

      <div className="relative flex w-full items-center space-x-2">
        <div className="relative flex-1">
          <div className={cn(
            "relative w-full border border-input rounded-xl bg-background shadow-sm overflow-hidden",
            className,
          )}>
            <textarea
              aria-label="Write your prompt here"
              placeholder={placeholder}
              ref={textAreaRef}
              onPaste={onPaste}
              onKeyDown={onKeyDown}
              className={cn(
                "z-10 w-full grow resize-none border-0 bg-transparent p-3 pr-20 text-base outline-none placeholder:text-muted-foreground text-foreground disabled:cursor-not-allowed disabled:opacity-50",
                showFileList && enableAdvancedFeatures && "pb-16",
                showFileList && !enableAdvancedFeatures && "pb-12", 
                !showFileList && enableAdvancedFeatures && "pb-10",
              )}
              {...(props.allowAttachments
                ? omit(props, ["allowAttachments", "files", "setFiles", "enableAdvancedFeatures"] as (keyof typeof props)[])
                : omit(props, ["allowAttachments", "enableAdvancedFeatures"] as (keyof typeof props)[]))}
            />
            
            {enableAdvancedFeatures && (
              <div className="px-3 py-2 flex items-center justify-between relative z-20">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <button
                    type="button"
                    onClick={() => setSearchEnabled(!searchEnabled)}
                    className={cn(
                      "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
                      searchEnabled
                        ? "bg-primary/10 text-primary hover:bg-primary/20"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    )}
                  >
                    <Search className="w-3.5 h-3.5" />
                    <span>Search</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setDeepResearchEnabled(!deepResearchEnabled)}
                    className={cn(
                      "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
                      deepResearchEnabled
                        ? "bg-primary/10 text-primary hover:bg-primary/20"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    )}
                  >
                    <Telescope className="w-3.5 h-3.5" />
                    <span>Deep Research</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setReasonEnabled(!reasonEnabled)}
                    className={cn(
                      "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
                      reasonEnabled
                        ? "bg-primary/10 text-primary hover:bg-primary/20"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    )}
                  >
                    <Zap className="w-3.5 h-3.5" />
                    <span>Reason</span>
                  </button>
                </div>
                {props.allowAttachments && (
                  <button
                    type="button"
                    onClick={async () => {
                      const files = await showFileUploadDialog();
                      addFiles(files);
                    }}
                    className="flex items-center gap-1.5 text-muted-foreground text-xs hover:text-foreground transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Upload Files</span>
                  </button>
                )}
              </div>
            )}
          </div>

          {props.allowAttachments && (
            <div className="absolute inset-x-3 bottom-0 z-20 overflow-x-auto py-3">
              <div className="flex space-x-3">
                <AnimatePresence mode="popLayout">
                  {props.files?.map((file) => {
                    return (
                      <FilePreview
                        key={file.name + String(file.lastModified)}
                        file={file}
                        onRemove={() => {
                          props.setFiles((files) => {
                            if (!files) return null;

                            const filtered = Array.from(files).filter(
                              (f) => f !== file,
                            );
                            if (filtered.length === 0) return null;
                            return filtered;
                          });
                        }}
                      />
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="absolute top-3 right-3 z-10 flex gap-2">
        {!enableAdvancedFeatures && props.allowAttachments && (
          <Button
            type="button"
            size="icon"
            variant="outline"
            className="h-8 w-8"
            aria-label="Attach a file"
            onClick={async () => {
              const files = await showFileUploadDialog();
              addFiles(files);
            }}
          >
            <Paperclip className="h-4 w-4" />
          </Button>
        )}
        {isSpeechSupported && (
          <button
            type="button"
            className="p-2 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Voice input"
            onClick={toggleListening}
          >
            <Mic className="h-4 w-4" />
          </button>
        )}
        {isGenerating && stop ? (
          <Button
            type="button"
            size="icon"
            className={cn(
              "h-8 w-8 rounded-full",
              props.value.trim() ? "bg-primary text-primary-foreground hover:bg-primary/90" : "bg-muted text-muted-foreground"
            )}
            aria-label="Stop generating"
            onClick={stop}
          >
            <Square className="h-3 w-3 animate-pulse" fill="currentColor" />
          </Button>
        ) : (
          <Button
            type="submit"
            size="icon"
            className={cn(
              "h-8 w-8 rounded-full transition-all",
              props.value.trim()
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
            aria-label="Send message"
            disabled={props.value === "" || isGenerating}
          >
            <ArrowUp className="h-4 w-4" />
          </Button>
        )}
      </div>

      {props.allowAttachments && <FileUploadOverlay isDragging={isDragging} />}

      <RecordingControls
        isRecording={isRecording}
        isTranscribing={isTranscribing}
        audioStream={audioStream}
        textAreaHeight={textAreaHeight}
        onStopRecording={stopRecording}
      />
    </div>
  );
}
MessageInput.displayName = "MessageInput";

interface FileUploadOverlayProps {
  isDragging: boolean;
}

function showFileUploadDialog() {
  return new Promise<File[] | null>((resolve, reject) => {
    try {
      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.accept = "*/*";

      // Set up event handlers before clicking
      input.onchange = (e) => {
        try {
          const files = (e.currentTarget as HTMLInputElement).files;

          if (files && files.length > 0) {
            resolve(Array.from(files));
          } else {
            resolve(null);
          }
        } catch (error) {
          reject(
            new Error(
              `Failed to access selected files: ${error instanceof Error ? error.message : "Unknown error"}`,
            ),
          );
        }
      };

      input.onerror = () => {
        reject(new Error("File input dialog encountered an error"));
      };

      input.oncancel = () => {
        resolve(null);
      };

      // Trigger the file dialog
      input.click();
    } catch (error) {
      reject(
        new Error(
          `Failed to create file dialog: ${error instanceof Error ? error.message : "Unknown error"}`,
        ),
      );
    }
  });
}

function FileUploadOverlay({ isDragging }: FileUploadOverlayProps) {
  return (
    <AnimatePresence>
      {isDragging && (
        <motion.div
          className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center space-x-2 rounded-xl border border-border border-dashed bg-background text-muted-foreground text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          aria-hidden
        >
          <Paperclip className="h-4 w-4" />
          <span>Drop your files here to attach them.</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function TranscribingOverlay() {
  return (
    <motion.div
      className="flex h-full w-full flex-col items-center justify-center rounded-xl bg-background/80 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <div className="mb-3">
        <TextShimmerLoader text="Transcribing audio" size="sm" />
      </div>
    </motion.div>
  );
}

interface RecordingPromptProps {
  isVisible: boolean;
  onStopRecording: () => void;
}

function RecordingPrompt({ isVisible, onStopRecording }: RecordingPromptProps) {
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ top: 0, filter: "blur(5px)" }}
          animate={{
            top: -40,
            filter: "blur(0px)",
            transition: {
              type: "spring",
              filter: { type: "tween" },
            },
          }}
          exit={{ top: 0, filter: "blur(5px)" }}
          className="-translate-x-1/2 absolute left-1/2 flex cursor-pointer overflow-hidden whitespace-nowrap rounded-full border bg-background py-1 text-center text-muted-foreground text-sm"
          onClick={onStopRecording}
        >
          <span className="mx-2.5 flex items-center">
            <Info className="mr-2 h-3 w-3" />
            Click to finish recording
          </span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

interface RecordingControlsProps {
  isRecording: boolean;
  isTranscribing: boolean;
  audioStream: MediaStream | null;
  textAreaHeight: number;
  onStopRecording: () => void;
}

function RecordingControls({
  isRecording,
  isTranscribing,
  audioStream,
  textAreaHeight,
  onStopRecording,
}: RecordingControlsProps) {
  if (isRecording) {
    return (
      <div
        className="absolute inset-[1px] z-50 overflow-hidden rounded-xl"
        style={{ height: textAreaHeight - 2 }}
      >
        <AudioVisualizer
          stream={audioStream}
          isRecording={isRecording}
          onClick={onStopRecording}
        />
      </div>
    );
  }

  if (isTranscribing) {
    return (
      <div
        className="absolute inset-[1px] z-50 overflow-hidden rounded-xl"
        style={{ height: textAreaHeight - 2 }}
      >
        <TranscribingOverlay />
      </div>
    );
  }

  return null;
}
