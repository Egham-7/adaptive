/**
 * Authentication-related types
 */

export type AuthResult =
	| {
			authType: "api_key";
			apiKey: {
				id: string;
				projectId: string;
				keyPrefix: string;
				project: { id: string; name: string };
			};
			project: { id: string; name: string };
	  }
	| {
			authType: "user";
			userId: string;
			project: { id: string; name: string };
	  };
