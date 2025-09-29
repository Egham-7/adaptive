/**
 * Standard API response types for Next.js API routes
 */

export interface JsonErrorResponse {
	error: {
		code: number;
		message: string;
		status: string;
		details: string;
	};
}

export interface ApiSuccessResponse<T = unknown> {
	data: T;
	success: true;
}

export interface ApiErrorResponse {
	error: {
		code: number;
		message: string;
		status: string;
		details?: string;
	};
	success: false;
}

export type ApiResponse<T = unknown> = ApiSuccessResponse<T> | ApiErrorResponse;
