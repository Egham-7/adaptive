import { ChatbotSidebar } from "@/components/chatbot-sidebar";

export function HomeLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background flex w-full">
      <ChatbotSidebar />
      <main className="flex-1 container px-10 pt-10 py-2 w-full overflow-hidden">
        {children}
      </main>
    </div>
  );
}
