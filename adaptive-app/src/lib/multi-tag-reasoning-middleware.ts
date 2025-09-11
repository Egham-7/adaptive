import type {
	LanguageModelV2Content,
	LanguageModelV2Middleware,
	LanguageModelV2StreamPart,
} from "@ai-sdk/provider";

// Interface for extraction state management
interface ExtractionState {
	detectedTagName?: string;
	isFirstReasoning: boolean;
	isFirstText: boolean;
	afterSwitch: boolean;
	isReasoning: boolean;
	buffer: string;
	idCounter: number;
	textId: string;
}

// Interface for compiled pattern structure
interface CompiledPattern {
	tagName: string;
	openingTag: string;
	closingTag: string;
	regex: RegExp;
}

// Helper function to escape regex special characters for security
function escapeRegex(string: string): string {
	return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Stream processor class to handle complex streaming logic
class StreamProcessor {
	constructor(
		private patterns: CompiledPattern[],
		private separator: string,
		private startWithReasoning: boolean,
	) {}

	createExtractionState(textId: string): ExtractionState {
		const firstPattern = this.patterns[0];
		if (!firstPattern && this.startWithReasoning) {
			console.warn("No patterns available for startWithReasoning");
		}

		return {
			isFirstReasoning: true,
			isFirstText: true,
			afterSwitch: false,
			isReasoning: this.startWithReasoning,
			buffer:
				this.startWithReasoning && firstPattern
					? `<${firstPattern.tagName}>`
					: "",
			idCounter: 0,
			textId,
		};
	}

	detectTagPattern(extraction: ExtractionState): boolean {
		if (extraction.detectedTagName) return true;

		for (const pattern of this.patterns) {
			if (extraction.buffer.includes(pattern.openingTag)) {
				extraction.detectedTagName = pattern.tagName;
				return true;
			}
		}
		return false;
	}

	publishContent(
		text: string,
		extraction: ExtractionState,
		controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
	): void {
		if (text.length === 0) return;

		const prefix =
			extraction.afterSwitch &&
			(extraction.isReasoning
				? !extraction.isFirstReasoning
				: !extraction.isFirstText)
				? this.separator
				: "";

		if (
			extraction.isReasoning &&
			(extraction.afterSwitch || extraction.isFirstReasoning)
		) {
			controller.enqueue({
				type: "reasoning-start",
				id: `reasoning-${extraction.idCounter}`,
			});
		}

		controller.enqueue(
			extraction.isReasoning
				? {
						type: "reasoning-delta",
						delta: prefix + text,
						id: `reasoning-${extraction.idCounter}`,
					}
				: {
						type: "text-delta",
						delta: prefix + text,
						id: extraction.textId,
					},
		);

		extraction.afterSwitch = false;

		if (extraction.isReasoning) {
			extraction.isFirstReasoning = false;
		} else {
			extraction.isFirstText = false;
		}
	}
}

// Helper function to find potential start index of a tag (from AI SDK source)
function getPotentialStartIndex(
	text: string,
	searchedText: string,
): number | null {
	if (searchedText.length === 0) {
		return null;
	}
	const directIndex = text.indexOf(searchedText);
	if (directIndex !== -1) {
		return directIndex;
	}
	// Look for partial matches at the end of the text
	for (let i = 1; i < searchedText.length && i <= text.length; i++) {
		if (text.endsWith(searchedText.substring(0, i))) {
			return text.length - i;
		}
	}
	return null;
}

/**
 * Extract reasoning from multiple possible XML tag patterns, useful for adaptive providers
 * where the model (and its reasoning tag format) is determined at runtime.
 *
 * This middleware supports both streaming and non-streaming responses, automatically
 * detecting and extracting reasoning content from XML-like tags while preserving
 * the remaining text content.
 *
 * @param options - Configuration options for the middleware
 * @param options.tagPatterns - Array of tag names to search for or a RegExp pattern.
 *   Defaults to ["think", "reasoning", "analysis", "thought"]. If a RegExp is provided,
 *   it will be used directly for pattern matching.
 * @param options.separator - String to join multiple reasoning sections. Defaults to "\n".
 * @param options.startWithReasoning - Whether to automatically wrap initial content
 *   in reasoning tags. Defaults to false. Useful for models that start responses
 *   with reasoning but don't include opening tags.
 *
 * @returns LanguageModelV2Middleware instance that processes both streaming and
 *   non-streaming responses
 *
 * @example
 * ```typescript
 * const middleware = multiTagReasoningMiddleware({
 *   tagPatterns: ["think", "reasoning"],
 *   separator: "\n\n",
 *   startWithReasoning: false
 * });
 *
 * const model = wrapLanguageModel({
 *   model: yourModel,
 *   middleware
 * });
 * ```
 */
export function multiTagReasoningMiddleware({
	tagPatterns = ["think", "reasoning", "analysis", "thought"],
	separator = "\n",
	startWithReasoning = false,
}: {
	tagPatterns?: string[] | RegExp;
	separator?: string;
	startWithReasoning?: boolean;
} = {}): LanguageModelV2Middleware {
	// If regex is provided, use it directly
	if (tagPatterns instanceof RegExp) {
		return createRegexReasoningMiddleware(tagPatterns, separator);
	}

	// For string patterns, try each one in order until we find a match
	const patterns = Array.isArray(tagPatterns) ? tagPatterns : [tagPatterns];

	// Pre-compile regex patterns for better performance
	const compiledPatterns: CompiledPattern[] = patterns.map((tagName) => ({
		tagName,
		openingTag: `<${tagName}>`,
		closingTag: `</${tagName}>`,
		regex: new RegExp(
			`${escapeRegex(`<${tagName}>`)}(.*?)${escapeRegex(`</${tagName}>`)}`,
			"gs",
		),
	}));

	// Create stream processor for handling complex streaming logic
	const streamProcessor = new StreamProcessor(
		compiledPatterns,
		separator,
		startWithReasoning,
	);

	return {
		middlewareVersion: "v2",
		wrapGenerate: async ({ doGenerate }) => {
			const { content, ...rest } = await doGenerate();

			const transformedContent: LanguageModelV2Content[] = [];

			for (const part of content) {
				if (part.type !== "text") {
					transformedContent.push(part);
					continue;
				}

				const firstPattern = patterns[0];
				const text =
					startWithReasoning && firstPattern
						? `<${firstPattern}>${part.text}</${firstPattern}>`
						: part.text;
				let foundMatch = false;

				// Try each compiled pattern until we find one that matches
				for (const pattern of compiledPatterns) {
					// Reset regex state for global flag
					pattern.regex.lastIndex = 0;
					const matches = Array.from(text.matchAll(pattern.regex));

					if (matches.length > 0) {
						const reasoningText = matches
							.map((match) => match[1]?.trim() || "")
							.filter((text) => text.length > 0)
							.join(separator);

						let textWithoutReasoning = text;
						for (let i = matches.length - 1; i >= 0; i--) {
							const match = matches[i];
							if (!match || match.index === undefined || match.index < 0) {
								console.warn("Invalid match detected, skipping");
								continue;
							}

							// Add bounds checking
							if (match.index >= textWithoutReasoning.length) {
								console.warn("Match index out of bounds, skipping");
								continue;
							}

							const endIndex = match.index + match[0].length;
							if (endIndex > textWithoutReasoning.length) {
								console.warn("Match end index out of bounds, skipping");
								continue;
							}

							const beforeMatch = textWithoutReasoning.slice(0, match.index);
							const afterMatch = textWithoutReasoning.slice(endIndex);

							textWithoutReasoning =
								beforeMatch +
								(beforeMatch.length > 0 && afterMatch.length > 0
									? separator
									: "") +
								afterMatch;
						}

						transformedContent.push({
							type: "reasoning",
							text: reasoningText,
						});

						if (textWithoutReasoning.trim()) {
							transformedContent.push({
								type: "text",
								text: textWithoutReasoning.trim(),
							});
						}

						foundMatch = true;
						break;
					}
				}

				if (!foundMatch) {
					transformedContent.push(part);
				}
			}

			return { content: transformedContent, ...rest };
		},

		wrapStream: async ({ doStream }) => {
			const { stream, ...rest } = await doStream();

			// For streaming, we'll detect the tag pattern dynamically as content arrives
			const reasoningExtractions: Record<string, ExtractionState> = {};

			return {
				stream: stream.pipeThrough(
					new TransformStream<
						LanguageModelV2StreamPart,
						LanguageModelV2StreamPart
					>({
						transform: (chunk, controller) => {
							if (chunk.type !== "text-delta") {
								controller.enqueue(chunk);
								return;
							}

							if (reasoningExtractions[chunk.id] == null) {
								reasoningExtractions[chunk.id] =
									streamProcessor.createExtractionState(chunk.id);
							}

							const activeExtraction = reasoningExtractions[chunk.id];
							if (!activeExtraction) {
								console.warn(
									"Active extraction not found for chunk ID:",
									chunk.id,
								);
								return;
							}

							activeExtraction.buffer += chunk.delta;

							// Auto-detect tag pattern if not already detected
							if (!streamProcessor.detectTagPattern(activeExtraction)) {
								controller.enqueue(chunk);
								return;
							}

							const openingTag = `<${activeExtraction.detectedTagName}>`;
							const closingTag = `</${activeExtraction.detectedTagName}>`;

							const publish = (text: string) => {
								streamProcessor.publishContent(
									text,
									activeExtraction,
									controller,
								);
							};

							do {
								const nextTag = activeExtraction.isReasoning
									? closingTag
									: openingTag;

								const startIndex = getPotentialStartIndex(
									activeExtraction.buffer,
									nextTag,
								);

								if (startIndex == null) {
									publish(activeExtraction.buffer);
									activeExtraction.buffer = "";
									break;
								}

								publish(activeExtraction.buffer.slice(0, startIndex));

								const foundFullMatch =
									startIndex + nextTag.length <= activeExtraction.buffer.length;

								if (foundFullMatch) {
									activeExtraction.buffer = activeExtraction.buffer.slice(
										startIndex + nextTag.length,
									);

									if (activeExtraction.isReasoning) {
										controller.enqueue({
											type: "reasoning-end",
											id: `reasoning-${activeExtraction.idCounter++}`,
										});
									}

									activeExtraction.isReasoning = !activeExtraction.isReasoning;
									activeExtraction.afterSwitch = true;
								} else {
									activeExtraction.buffer =
										activeExtraction.buffer.slice(startIndex);
									break;
								}
							} while (activeExtraction.buffer.length > 0);
						},
						flush: () => {
							// Cleanup memory to prevent leaks
							Object.keys(reasoningExtractions).forEach((key) => {
								delete reasoningExtractions[key];
							});
						},
					}),
				),
				...rest,
			};
		},
	};
}

/**
 * Helper function for regex-based reasoning extraction.
 * Creates middleware that uses a single RegExp pattern for reasoning extraction.
 *
 * @param pattern - RegExp pattern to match reasoning content
 * @param separator - String to join multiple reasoning sections
 * @returns LanguageModelV2Middleware instance
 */
function createRegexReasoningMiddleware(
	pattern: RegExp,
	separator: string,
): LanguageModelV2Middleware {
	return {
		middlewareVersion: "v2",
		wrapGenerate: async ({ doGenerate }) => {
			const { content, ...rest } = await doGenerate();

			const transformedContent: LanguageModelV2Content[] = [];

			for (const part of content) {
				if (part.type !== "text") {
					transformedContent.push(part);
					continue;
				}

				const matches = Array.from(part.text.matchAll(pattern));

				if (matches.length === 0) {
					transformedContent.push(part);
					continue;
				}

				const reasoningText = matches
					.map((match) => match[1] || match[0])
					.join(separator);
				const textWithoutReasoning = part.text.replace(pattern, "").trim();

				transformedContent.push({
					type: "reasoning",
					text: reasoningText,
				});

				if (textWithoutReasoning) {
					transformedContent.push({
						type: "text",
						text: textWithoutReasoning,
					});
				}
			}

			return { content: transformedContent, ...rest };
		},

		wrapStream: async ({ doStream }) => {
			// For regex patterns in streaming, we'd need more complex logic
			// This is a simplified version - full implementation would buffer
			// and apply regex at appropriate boundaries
			const { stream, ...rest } = await doStream();
			return { stream, ...rest };
		},
	};
}
