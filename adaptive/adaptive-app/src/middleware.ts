import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isProtectedRoute = createRouteMatcher([
	"/chat-platform(.*)",
	"/api-platform(.*)",
]);

export default clerkMiddleware(async (auth, req) => {
	if (isProtectedRoute(req)) await auth.protect();
});

export const config = {
	matcher: [
		// Exclude `/api/v1` routes explicitly
		"/((?!.swa|_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)|api/stripe|api/v1/.*).*)",

		// This line ensures API routes (except v1) are always processed.
		"/(api(?!/v1)|trpc)(.*)",
	],
};
