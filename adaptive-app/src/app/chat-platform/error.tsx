"use client";

import { AlertTriangle, MessageCircleX, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const router = useRouter();

  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Conversation creation failed:", error);
  }, [error]);

  const isDevelopment = process.env.NODE_ENV === "development";

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-lg">
        {/* Error Icon */}
        <div className="mb-6 flex justify-center">
          <div className="relative">
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-destructive/10">
              <MessageCircleX className="h-10 w-10 text-destructive" />
            </div>
            <div className="-bottom-1 -right-1 absolute flex h-6 w-6 items-center justify-center rounded-full bg-destructive">
              <AlertTriangle className="h-3 w-3 text-destructive-foreground" />
            </div>
          </div>
        </div>

        {/* Error Content */}
        <div className="space-y-4 text-center">
          <h1 className="font-bold text-2xl text-foreground">
            Couldn't Start Conversation
          </h1>

          <p className="text-muted-foreground leading-relaxed">
            We're having trouble creating your new conversation right now. This
            might be due to a temporary server issue or network problem.
          </p>

          {/* Quick Tips */}
          <div className="mt-6 rounded-lg border border-border bg-muted/50 p-4">
            <h3 className="mb-3 font-medium text-foreground text-sm">
              Quick fixes to try:
            </h3>
            <ul className="space-y-2 text-left text-muted-foreground text-sm">
              <li className="flex items-start gap-2">
                <span className="mt-2 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
                Check your internet connection
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-2 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
                Try refreshing the page
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-2 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
                Clear your browser cache
              </li>
            </ul>
          </div>

          {/* Error Details (Development Only) */}
          {isDevelopment && error.message && (
            <div className="mt-6 rounded-lg border border-border bg-muted p-4">
              <div className="mb-2 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium text-muted-foreground text-sm">
                  Debug Info
                </span>
              </div>
              <p className="break-all rounded border bg-card p-3 font-mono text-foreground text-sm">
                {error.message}
              </p>
              {error.digest && (
                <p className="mt-2 text-muted-foreground text-xs">
                  Error ID: {error.digest}
                </p>
              )}
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-8 flex flex-col gap-3">
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-6 py-3 font-medium text-primary-foreground shadow-sm transition-colors hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              <RefreshCw className="h-4 w-4" />
              Try Creating Conversation Again
            </button>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => router.push("/")}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-secondary px-4 py-2.5 font-medium text-secondary-foreground transition-colors hover:bg-secondary/80 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              >
                Back to Home
              </button>

              <button
                type="button"
                onClick={() => router.refresh()}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg border border-border px-4 py-2.5 font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              >
                Refresh Page
              </button>
            </div>
          </div>

          {/* Status Message */}
          <div className="mt-8 border-border border-t pt-6">
            <p className="text-muted-foreground text-sm">
              If this keeps happening, check our{" "}
              <a
                href="/status"
                className="text-primary underline underline-offset-4 hover:text-primary/80"
              >
                status page
              </a>{" "}
              or{" "}
              <a
                href="mailto:support@yourapp.com"
                className="text-primary underline underline-offset-4 hover:text-primary/80"
              >
                contact support
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
