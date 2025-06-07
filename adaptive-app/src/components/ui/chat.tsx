import { useDeleteMessage } from "@/hooks/messages/use-delete-message";
import {
	AlertTriangle,
	ArrowDown,
	Edit3,
	RotateCcw,
	ThumbsDown,
	ThumbsUp,
	Trash2,
} from "lucide-react";
import {
	type ReactElement,
	forwardRef,
	useCallback,
	useReducer,
	useRef,
	useState,
} from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { MessageInput } from "@/components/ui/message-input";
import { MessageList } from "@/components/ui/message-list";
import { PromptSuggestions } from "@/components/ui/prompt-suggestions";
import { useAutoScroll } from "@/hooks/use-auto-scroll";
import { cn } from "@/lib/utils";
import type { Message, UIMessage } from "ai";

// Infer specific part types from UIMessage for safety and clarity
type MessageTextPart = Extract<UIMessage["parts"][number], { type: "text" }>;
type MessageToolInvocationPart = Extract<
	UIMessage["parts"][number],
	{ type: "tool-invocation" }
>;

type MessageAction =
	| { type: "CANCEL_TOOL_INVOCATIONS"; messageId: string }
	| { type: "DELETE_MESSAGE_AND_AFTER"; messageId: string }
	| { type: "DELETE_MESSAGES_AFTER"; messageIndex: number }
	| { type: "SET_MESSAGES"; messages: Message[] }
	| { type: "EDIT_MESSAGE"; messageId: string; content: string }
	| {
			type: "RATE_MESSAGE";
			messageId: string;
			rating: "thumbs-up" | "thumbs-down";
	  }
	| { type: "RETRY_MESSAGE"; messageId: string; content: string }
	| { type: "ADD_MESSAGE"; message: Message };

interface MessageState {
	messages: Message[];
	editingMessageId: string | null;
	editingContent: string;
}

function messageReducer(
	state: MessageState,
	action: MessageAction,
): MessageState {
	switch (action.type) {
		case "CANCEL_TOOL_INVOCATIONS": {
			const messageIndex = state.messages.findIndex(
				(m) => m.id === action.messageId,
			);
			if (messageIndex === -1) return state;

			const message = state.messages[messageIndex] as UIMessage;
			if (!message.parts?.length) return state;

			let needsUpdate = false;
			const updatedParts = message.parts.map((part) => {
				if (
					part.type === "tool-invocation" &&
					(part as MessageToolInvocationPart).toolInvocation?.state === "call"
				) {
					needsUpdate = true;
					return {
						...part,
						toolInvocation: {
							...(part as MessageToolInvocationPart).toolInvocation,
							state: "result",
							result: {
								content: "Tool execution was cancelled",
								__cancelled: true,
							},
						},
					} as MessageToolInvocationPart;
				}
				return part;
			});

			if (!needsUpdate) return state;

			const newMessages = [...state.messages];
			newMessages[messageIndex] = {
				...message,
				parts: updatedParts,
			};

			return {
				...state,
				messages: newMessages,
			};
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

		case "DELETE_MESSAGES_AFTER": {
			return {
				...state,
				messages: state.messages.slice(0, action.messageIndex),
				editingMessageId: null,
				editingContent: "",
			};
		}

		case "SET_MESSAGES": {
			return {
				...state,
				messages: action.messages,
			};
		}

		case "EDIT_MESSAGE": {
			return {
				...state,
				editingMessageId: action.messageId,
				editingContent: action.content,
			};
		}

		case "RATE_MESSAGE": {
			// Note: This would typically update message metadata or trigger external API call
			// For now, we'll just return the current state as rating is handled externally
			return state;
		}

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

		case "ADD_MESSAGE": {
			return {
				...state,
				messages: [...state.messages, action.message],
			};
		}

		default:
			return state;
	}
}

interface ChatPropsBase {
	handleSubmit: (
		event?: { preventDefault?: () => void },
		options?: { experimental_attachments?: FileList },
	) => void;
	messages: Array<UIMessage>;
	input: string;
	className?: string;
	handleInputChange: React.ChangeEventHandler<HTMLTextAreaElement>;
	isGenerating: boolean;
	stop?: () => void;
	onRateResponse?: (
		messageId: string,
		rating: "thumbs-up" | "thumbs-down",
	) => void;
	setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
	transcribeAudio?: (blob: Blob) => Promise<string>;
	isError?: boolean;
	error?: Error;
	onRetry?: () => void;
}

interface ChatPropsWithoutSuggestions extends ChatPropsBase {
	append?: never;
	suggestions?: never;
}

interface ChatPropsWithSuggestions extends ChatPropsBase {
	append: (message: { role: "user"; content: string }) => void;
	suggestions: string[];
}

type ChatProps = ChatPropsWithoutSuggestions | ChatPropsWithSuggestions;

export function Chat({
	messages,
	handleSubmit,
	input,
	handleInputChange,
	stop,
	isGenerating,
	append,
	suggestions,
	className,
	onRateResponse,
	setMessages,
	transcribeAudio,
	isError,
	error,
	onRetry,
}: ChatProps) {
	const deleteMessageMutation = useDeleteMessage();
	const [state, dispatch] = useReducer(messageReducer, {
		messages,
		editingMessageId: null,
		editingContent: "",
	});

	// Sync external messages with internal state
	const messagesRef = useRef(messages);
	if (messagesRef.current !== messages) {
		messagesRef.current = messages;
		dispatch({ type: "SET_MESSAGES", messages });
	}

	// Sync internal state changes back to external setMessages
	const internalMessagesRef = useRef(state.messages);
	if (internalMessagesRef.current !== state.messages) {
		internalMessagesRef.current = state.messages;
		setMessages(state.messages);
	}

	const lastMessage = state.messages.at(-1);
	const isEmpty = state.messages.length === 0;
	const isTyping = lastMessage?.role === "user";

	const handleStop = useCallback(() => {
		stop?.();

		const lastAssistantMessage = state.messages.findLast(
			(m: Message) => m.role === "assistant",
		);

		if (lastAssistantMessage) {
			dispatch({
				type: "CANCEL_TOOL_INVOCATIONS",
				messageId: lastAssistantMessage.id,
			});
		}
	}, [stop, state.messages]);

	const handleEditMessage = useCallback(
		(messageId: string, content: string) => {
			dispatch({
				type: "EDIT_MESSAGE",
				messageId,
				content,
			});
		},
		[],
	);

	const handleSaveEdit = useCallback(
		(messageId: string) => {
			if (!state.editingContent.trim()) return;

			const messageIndex = state.messages.findIndex((m) => m.id === messageId);
			if (messageIndex !== -1) {
				// Delete subsequent messages from database
				const messagesToDelete = state.messages.slice(messageIndex);
				messagesToDelete.forEach((msg) => {
					deleteMessageMutation.mutate({ id: msg.id });
				});

				setMessages(messages.slice(0, messageIndex));
				append?.({
					role: "user",
					content: state.editingContent.trim(),
				});

				// Clear editing state
				dispatch({ type: "EDIT_MESSAGE", messageId: "", content: "" });
			}
		},
		[state.messages, state.editingContent, deleteMessageMutation, setMessages],
	);

	const handleCancelEdit = useCallback(() => {
		dispatch({
			type: "EDIT_MESSAGE",
			messageId: "",
			content: "",
		});
	}, []);

	const handleRetryMessage = useCallback(
		(message: UIMessage) => {
			if (!append) return;

			const messageIndex = state.messages.findIndex((m) => m.id === message.id);
			if (messageIndex !== -1) {
				// Delete subsequent messages from database
				const messagesToDelete = state.messages.slice(messageIndex);
				messagesToDelete.forEach((msg) => {
					deleteMessageMutation.mutate({ id: msg.id });
				});
			}

			dispatch({
				type: "RETRY_MESSAGE",
				messageId: message.id,
				content: message.content,
			});

			// Re-send the message
			append({ role: "user", content: message.content });
		},
		[append, state.messages, deleteMessageMutation],
	);

	const handleDeleteMessage = useCallback(
		(messageId: string) => {
			const messageIndex = state.messages.findIndex((m) => m.id === messageId);
			if (messageIndex !== -1) {
				// Delete this message and all subsequent messages from database
				const messagesToDelete = state.messages.slice(messageIndex);
				messagesToDelete.forEach((msg) => {
					deleteMessageMutation.mutate({ id: msg.id });
				});
			}

			dispatch({
				type: "DELETE_MESSAGE_AND_AFTER",
				messageId,
			});
		},
		[state.messages, deleteMessageMutation],
	);

	const handleRateMessage = useCallback(
		(messageId: string, rating: "thumbs-up" | "thumbs-down") => {
			dispatch({
				type: "RATE_MESSAGE",
				messageId,
				rating,
			});

			// Call external rating handler
			onRateResponse?.(messageId, rating);
		},
		[onRateResponse],
	);

	const messageOptions = useCallback(
		(message: UIMessage) => {
			const isUserMessage = message.role === "user";
			const canEdit = isUserMessage && !isGenerating;
			const canRetry = isUserMessage && !isGenerating;
			const canDelete = !isGenerating;

			if (isUserMessage) {
				return {
					actions: (
						<>
							{canEdit && (
								<Button
									size="icon"
									variant="ghost"
									className="h-6 w-6"
									onClick={() => handleEditMessage(message.id, message.content)}
									disabled={state.editingMessageId === message.id}
								>
									<Edit3 className="h-4 w-4" />
								</Button>
							)}
							{canRetry && (
								<Button
									size="icon"
									variant="ghost"
									className="h-6 w-6"
									onClick={() => handleRetryMessage(message)}
								>
									<RotateCcw className="h-4 w-4" />
								</Button>
							)}
							{canDelete && (
								<Button
									size="icon"
									variant="ghost"
									className="h-6 w-6 text-destructive hover:text-destructive"
									onClick={() => handleDeleteMessage(message.id)}
								>
									<Trash2 className="h-4 w-4" />
								</Button>
							)}
						</>
					),
					isEditing: state.editingMessageId === message.id,
					editingContent: state.editingContent,
					onEditingContentChange: (content: string) =>
						dispatch({
							type: "EDIT_MESSAGE",
							messageId: state.editingMessageId || "",
							content,
						}),
					onSaveEdit: () => handleSaveEdit(message.id),
					onCancelEdit: handleCancelEdit,
				};
			}

			// Assistant message actions
			return {
				actions: onRateResponse ? (
					<>
						<div className="border-r pr-1">
							<CopyButton
								content={
									(
										message.parts?.find(
											(p) => p.type === "text",
										) as MessageTextPart
									)?.text || message.content
								}
								copyMessage="Copied response to clipboard!"
							/>
						</div>
						<Button
							size="icon"
							variant="ghost"
							className="h-6 w-6"
							onClick={() => handleRateMessage(message.id, "thumbs-up")}
						>
							<ThumbsUp className="h-4 w-4" />
						</Button>
						<Button
							size="icon"
							variant="ghost"
							className="h-6 w-6"
							onClick={() => handleRateMessage(message.id, "thumbs-down")}
						>
							<ThumbsDown className="h-4 w-4" />
						</Button>
					</>
				) : (
					<CopyButton
						content={
							(message.parts?.find((p) => p.type === "text") as MessageTextPart)
								?.text || message.content
						}
						copyMessage="Copied response to clipboard!"
					/>
				),
			};
		},
		[
			onRateResponse,
			isGenerating,
			state.editingMessageId,
			state.editingContent,
			handleEditMessage,
			handleSaveEdit,
			handleCancelEdit,
			handleRetryMessage,
			handleDeleteMessage,
			handleRateMessage,
		],
	);

	return (
		<ChatContainer className={className}>
			{isEmpty && append && suggestions ? (
				<PromptSuggestions
					label="Try these prompts ✨"
					append={append}
					suggestions={suggestions}
				/>
			) : null}

			{state.messages.length > 0 ? (
				<ChatMessages messages={state.messages as UIMessage[]}>
					<MessageList
						messages={state.messages as UIMessage[]}
						isTyping={isTyping}
						messageOptions={messageOptions}
					/>
				</ChatMessages>
			) : null}

			{isError && <ChatErrorDisplay error={error} onRetry={onRetry} />}

			<ChatForm
				className="mt-auto"
				isPending={isGenerating || isTyping}
				handleSubmit={handleSubmit}
			>
				{({ files, setFiles }) => (
					<MessageInput
						value={input}
						onChange={handleInputChange}
						allowAttachments
						files={files}
						setFiles={setFiles}
						stop={handleStop}
						isGenerating={isGenerating}
						transcribeAudio={transcribeAudio}
					/>
				)}
			</ChatForm>
		</ChatContainer>
	);
}

function ChatErrorDisplay({
	error,
	onRetry,
}: {
	error?: Error;
	onRetry?: () => void;
}) {
	if (!error) return null;

	return (
		<div className=" w-full max-w-2xl px-4 pb-4">
			<Alert variant="destructive">
				<AlertTriangle className="h-4 w-4" />
				<AlertTitle>Error</AlertTitle>
				<AlertDescription>
					<p>{error.message || "An unexpected error occurred."}</p>
					{onRetry && (
						<Button
							variant="secondary"
							size="sm"
							onClick={onRetry}
							className="mt-2"
						>
							Retry
						</Button>
					)}
				</AlertDescription>
			</Alert>
		</div>
	);
}

export function ChatMessages({
	messages,
	children,
}: React.PropsWithChildren<{
	messages: UIMessage[];
}>) {
	const {
		containerRef,
		scrollToBottom,
		handleScroll,
		shouldAutoScroll,
		handleTouchStart,
	} = useAutoScroll([messages]);

	return (
		<div
			className="grid grid-cols-1 overflow-y-auto pb-4"
			ref={containerRef}
			onScroll={handleScroll}
			onTouchStart={handleTouchStart}
		>
			<div className="max-w-full [grid-column:1/1] [grid-row:1/1]">
				{children}
			</div>

			{!shouldAutoScroll && (
				<div className="pointer-events-none flex flex-1 items-end justify-end [grid-column:1/1] [grid-row:1/1]">
					<div className="sticky bottom-0 left-0 flex w-full justify-end">
						<Button
							onClick={scrollToBottom}
							className="fade-in-0 slide-in-from-bottom-1 pointer-events-auto h-8 w-8 animate-in rounded-full ease-in-out"
							size="icon"
							variant="ghost"
						>
							<ArrowDown className="h-4 w-4" />
						</Button>
					</div>
				</div>
			)}
		</div>
	);
}

export const ChatContainer = forwardRef<
	HTMLDivElement,
	React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
	return (
		<div
			ref={ref}
			className={cn("grid max-h-full w-full grid-rows-[1fr_auto]", className)}
			{...props}
		/>
	);
});
ChatContainer.displayName = "ChatContainer";

interface ChatFormProps {
	className?: string;
	isPending: boolean;
	handleSubmit: (
		event?: { preventDefault?: () => void },
		options?: { experimental_attachments?: FileList },
	) => void;
	children: (props: {
		files: File[] | null;
		setFiles: React.Dispatch<React.SetStateAction<File[] | null>>;
	}) => ReactElement;
}

export const ChatForm = forwardRef<HTMLFormElement, ChatFormProps>(
	({ children, handleSubmit, className }, ref) => {
		const [files, setFiles] = useState<File[] | null>(null);

		const onSubmit = (event: React.FormEvent) => {
			if (!files) {
				handleSubmit(event);
				return;
			}

			const fileList = createFileList(files);
			handleSubmit(event, { experimental_attachments: fileList });
			setFiles(null);
		};

		return (
			<form ref={ref} onSubmit={onSubmit} className={className}>
				{children({ files, setFiles })}
			</form>
		);
	},
);
ChatForm.displayName = "ChatForm";

function createFileList(files: File[] | FileList): FileList {
	const dataTransfer = new DataTransfer();
	for (const file of Array.from(files)) {
		dataTransfer.items.add(file);
	}
	return dataTransfer.files;
}
