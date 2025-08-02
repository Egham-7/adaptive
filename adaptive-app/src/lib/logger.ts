/**
 * Centralized logging utility
 * Provides structured logging with different levels and optional Sentry integration
 */

import * as Sentry from "@sentry/nextjs";

export type LogLevel = "debug" | "info" | "warn" | "error";

interface LogContext {
	[key: string]: any;
}

class Logger {
	private isDevelopment = process.env.NODE_ENV === "development";

	private formatMessage(
		level: LogLevel,
		message: string,
		context?: LogContext,
	): string {
		const timestamp = new Date().toISOString();
		const contextStr = context ? ` | ${JSON.stringify(context)}` : "";
		return `[${timestamp}] ${level.toUpperCase()}: ${message}${contextStr}`;
	}

	debug(message: string, context?: LogContext) {
		if (this.isDevelopment) {
			console.log(this.formatMessage("debug", message, context));
		}
	}

	info(message: string, context?: LogContext) {
		const formatted = this.formatMessage("info", message, context);
		console.log(formatted);

		// Add breadcrumb for Sentry
		Sentry.addBreadcrumb({
			message,
			level: "info",
			data: context,
		});
	}

	warn(message: string, context?: LogContext) {
		const formatted = this.formatMessage("warn", message, context);
		console.warn(formatted);

		// Add breadcrumb for Sentry
		Sentry.addBreadcrumb({
			message,
			level: "warning",
			data: context,
		});
	}

	error(message: string, error?: Error, context?: LogContext) {
		const formatted = this.formatMessage("error", message, context);
		console.error(formatted);

		if (error) {
			console.error(error);
			// Report to Sentry
			Sentry.captureException(error, {
				extra: { message, ...context },
			});
		} else {
			// Report message to Sentry
			Sentry.captureMessage(message, "error");
		}
	}

	// Specialized methods for common operations
	transaction(operation: string, context?: LogContext) {
		this.info(`💳 ${operation}`, context);
	}

	credit(operation: string, context?: LogContext) {
		this.info(`💰 ${operation}`, context);
	}

	promotion(operation: string, context?: LogContext) {
		this.info(`🎉 ${operation}`, context);
	}

	success(operation: string, context?: LogContext) {
		this.info(`✅ ${operation}`, context);
	}

	failure(operation: string, context?: LogContext) {
		this.warn(`❌ ${operation}`, context);
	}
}

// Export singleton instance
export const logger = new Logger();
