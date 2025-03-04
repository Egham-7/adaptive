import { Outlet } from "@tanstack/react-router";
import { ModeToggle } from "@/components/mode-toggle";

export function HomeLayout() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-display text-xl font-bold">Adaptive</span>
          </div>
          <ModeToggle />
        </div>
      </header>

      {/* Main Content with Outlet */}
      <main className="flex min-h-screen w-full flex-col items-center justify-center pt-16 pb-8">
        <div className="w-full max-w-3xl space-y-8">
          <Outlet />
        </div>
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
  );
}
