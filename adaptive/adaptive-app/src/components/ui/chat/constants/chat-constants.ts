/**
 * Chat-related constants and configuration
 */

export const CHAT_LIMITS = {
  LOW_REMAINING_THRESHOLD: 5,
  WARNING_REMAINING_THRESHOLD: 2,
} as const;

export const MESSAGE_ROLES = {
  USER: "user",
  ASSISTANT: "assistant",
  SYSTEM: "system",
  TOOL: "tool",
} as const;

export const CHAT_STATUS = {
  NORMAL: "normal",
  LOW: "low", 
  WARNING: "warning",
  REACHED: "reached",
} as const;

export const RATING_TYPES = {
  THUMBS_UP: "thumbs-up",
  THUMBS_DOWN: "thumbs-down",
} as const;

export const ANIMATION_TYPES = {
  NONE: "none",
  SLIDE: "slide", 
  SCALE: "scale",
  FADE: "fade",
} as const;

export type ChatStatus = typeof CHAT_STATUS[keyof typeof CHAT_STATUS];
export type RatingType = typeof RATING_TYPES[keyof typeof RATING_TYPES];
export type AnimationType = typeof ANIMATION_TYPES[keyof typeof ANIMATION_TYPES];