import { ChatbotSidebar } from "@/components/chatbot-sidebar";
import { useAuth } from "@clerk/clerk-react";
import { HomeLayoutSkeleton } from "@/components/skeletons/home-layout-skeleton";
import { useRouter } from "next/navigation";

export function HomeLayout({ children }: { children: React.ReactNode }) {
  const { isLoaded, isSignedIn } = useAuth();

  const { push } = useRouter();

  if (!isLoaded) {
    return <HomeLayoutSkeleton />;
  }

  if (!isSignedIn) {
    push("/");
  }

  return (
    <div className="min-h-screen bg-background flex w-full">
      <ChatbotSidebar />
      <main className="flex-1 container px-10 pt-10 py-2 w-full overflow-hidden">
        {children}
      </main>
    </div>
  );
}
