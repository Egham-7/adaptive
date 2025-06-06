import { api } from "@/trpc/server";
import { ChatbotSidebar } from "../_components/chat-platform/chat-sidebar";

export default async function HomeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const initialConversations = await api.conversations.list();
  return (
    <div className="min-h-screen bg-background flex w-full">
      <ChatbotSidebar initialConversations={initialConversations} />
      <main className="flex-1 px-10 pt-10 py-2 w-full overflow-hidden">
        {children}
      </main>
    </div>
  );
}
