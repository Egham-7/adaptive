import { Outlet } from "@tanstack/react-router";
import { SidebarTrigger } from "@/components/ui/sidebar";
import ApiPlatformSidebar from "@/components/api-platform-sidebar";
import { useAuth } from "@clerk/clerk-react";
import { useRouter } from "@tanstack/react-router";
import { ApiPlatformSkeleton } from "@/components/skeletons/api-platform-layout-skeleton";

export function APIPlatformLayout() {
  const { isLoaded, isSignedIn } = useAuth();

  const { navigate } = useRouter();

  if (!isLoaded) {
    return <ApiPlatformSkeleton />;
  }

  if (!isSignedIn) {
    navigate({ to: "/" });
  }
  return (
    <div className="flex min-h-screen">
      <ApiPlatformSidebar />

      <div className="flex-1">
        <header className="flex h-14 items-center gap-4 border-b bg-background px-6 lg:h-[60px]">
          <SidebarTrigger />
          <div className="w-full flex-1">
            <h1 className="text-lg font-semibold">Adaptive Dashboard</h1>
          </div>
        </header>
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
