import { ModeToggle } from "@/components/mode-toggle";
import { Skeleton } from "@/components/ui/skeleton";

export function HomeLayoutSkeleton() {
  return (
    <div className="min-h-screen bg-background flex w-full">
      {/* Sidebar Skeleton */}
      <div className="hidden md:flex w-64 flex-col border-r">
        <div className="p-4 border-b">
          <Skeleton className="h-8 w-3/4" />
        </div>
        <div className="flex-1 overflow-auto p-4">
          <div className="space-y-3">
            {Array(6)
              .fill(0)
              .map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col px-10 w-full">
        {/* Header Skeleton */}
        <header className="sticky top-0 border-b bg-background/95 backdrop-blur-sm supports-backdrop-filter:bg-background/60 z-10">
          <div className="container flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <Skeleton className="h-6 w-6" />
              <Skeleton className="h-8 w-32" />
            </div>
            <ModeToggle />
          </div>
        </header>

        {/* Main Content Skeleton */}
        <main className="flex-1 container py-8">
          <div className="space-y-6">
            <Skeleton className="h-12 w-3/4" />
            <div className="grid gap-6">
              <Skeleton className="h-64 w-full rounded-lg" />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Skeleton className="h-32 w-full rounded-lg" />
                <Skeleton className="h-32 w-full rounded-lg" />
              </div>
            </div>
          </div>
        </main>

        {/* Footer Skeleton */}
        <footer className="border-t">
          <div className="container flex h-16 items-center justify-between">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-4 w-36" />
          </div>
        </footer>
      </div>
    </div>
  );
}
