import { Skeleton } from "@/components/ui/skeleton";

export function ApiPlatformSkeleton() {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar skeleton */}
      <div className="hidden w-64 flex-col border-r bg-background lg:flex">
        <div className="p-4 border-b">
          <Skeleton className="h-8 w-32" />
        </div>
        <div className="flex-1 p-4 space-y-4">
          {/* Menu items skeletons */}
          {Array(5)
            .fill(0)
            .map((_, i) => (
              <div key={i} className="flex items-center gap-3">
                <Skeleton className="h-5 w-5" />
                <Skeleton className="h-4 w-24" />
              </div>
            ))}
        </div>
        <div className="p-4 border-t">
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
      </div>

      {/* Main content skeleton */}
      <div className="flex-1">
        <header className="flex h-14 items-center gap-4 border-b bg-background px-6 lg:h-[60px]">
          <Skeleton className="h-6 w-6" />
          <div className="w-full flex-1">
            <Skeleton className="h-6 w-48" />
          </div>
        </header>

        <main className="p-6">
          <div className="space-y-6">
            <Skeleton className="h-8 w-64" />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array(3)
                .fill(0)
                .map((_, i) => (
                  <Skeleton key={i} className="h-40 w-full rounded-lg" />
                ))}
            </div>
            <Skeleton className="h-64 w-full rounded-lg" />
          </div>
        </main>
      </div>
    </div>
  );
}
