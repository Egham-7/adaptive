import { Outlet, useRouter } from "@tanstack/react-router";
import { ChatbotSidebar } from "@/components/chatbot-sidebar";
import { useAuth } from "@clerk/clerk-react";
import { HomeLayoutSkeleton } from "@/components/skeletons/home-layout-skeleton";

export function HomeLayout() {
  const { isLoaded, isSignedIn } = useAuth();

  const { navigate } = useRouter();

  if (!isLoaded) {
    return <HomeLayoutSkeleton />;
  }

  if (!isSignedIn) {
    navigate({ to: "/" });
  }

  return (
    <div className="min-h-screen bg-background flex w-full">
      <ChatbotSidebar />
      <main className="flex-1 container px-10 pt-10 py-2 w-full overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
