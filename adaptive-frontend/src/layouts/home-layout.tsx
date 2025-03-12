import { Outlet } from "@tanstack/react-router";
import { ModeToggle } from "@/components/mode-toggle";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { ChatbotSidebar } from "@/components/chatbot-sidebar";

export function HomeLayout() {
  const { open, openMobile } = useSidebar();
  return (
    <div className="min-h-screen bg-background flex w-full">
      <ChatbotSidebar />
      <div className="flex-1 flex flex-col px-10 w-full">
        {/* Header */}
        <header className="sticky top-0 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
          <div className="container flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              {!openMobile && !open && <SidebarTrigger />}
              <span className="font-display text-xl font-bold">Adaptive</span>
            </div>
            <ModeToggle />
          </div>
        </header>

        {/* Main Content with Outlet */}
        <main className="flex-1 container py-8">
          <Outlet />
        </main>

        {/* Footer */}
        <footer className="border-t">
          <div className="container flex h-16 items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Built with modern technologies
            </p>
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} Adaptive AI
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
