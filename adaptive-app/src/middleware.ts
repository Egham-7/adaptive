import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isProtectedRoute = createRouteMatcher(["/chat-platform(.*)"]);

export default clerkMiddleware(async (auth, req) => {
	if (isProtectedRoute(req)) await auth.protect();
});

export const config = {
	matcher: [
		// This first regex is the one we are updating.
		// It now includes a negative lookahead for `.swa` to exclude Azure's health check routes.
		"/((?!.swa|_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)|api/stripe).*)",

		// This line remains the same, ensuring API routes are always processed.
		"/(api|trpc)(.*)",
	],
};
