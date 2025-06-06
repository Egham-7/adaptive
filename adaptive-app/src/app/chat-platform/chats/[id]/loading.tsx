export default function Loading() {
  return (
    <div className="flex h-full w-full flex-col bg-background">
      {/* Chat header skeleton */}
      <div className="border-b border-border p-4">
        <div className="h-6 w-48 animate-pulse rounded-lg bg-muted"></div>
      </div>

      {/* Messages area skeleton */}
      <div className="flex-1 space-y-4 p-4">
        {/* User message skeleton */}
        <div className="flex justify-end">
          <div className="max-w-xs rounded-lg bg-primary/10 p-3">
            <div className="h-4 w-32 animate-pulse rounded bg-muted"></div>
          </div>
        </div>

        {/* Assistant message skeleton */}
        <div className="flex justify-start">
          <div className="max-w-xs rounded-lg bg-card p-3 shadow-sm">
            <div className="space-y-2">
              <div className="h-4 w-48 animate-pulse rounded bg-muted"></div>
              <div className="h-4 w-36 animate-pulse rounded bg-muted"></div>
              <div className="h-4 w-24 animate-pulse rounded bg-muted"></div>
            </div>
          </div>
        </div>

        {/* Another user message skeleton */}
        <div className="flex justify-end">
          <div className="max-w-xs rounded-lg bg-primary/10 p-3">
            <div className="h-4 w-40 animate-pulse rounded bg-muted"></div>
          </div>
        </div>
      </div>

      {/* Input area skeleton */}
      <div className="border-t border-border p-4">
        <div className="h-12 w-full animate-pulse rounded-lg bg-input"></div>
      </div>
    </div>
  );
}
