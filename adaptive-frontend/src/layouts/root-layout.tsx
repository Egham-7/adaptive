import { Outlet } from "@tanstack/react-router";
import { ThemeProvider } from "@/context/theme-provider";
import { Toaster } from "@/components/ui/sonner";

export function RootLayout() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <main className="flex-1">
        <Outlet />
      </main>
      <Toaster />
    </ThemeProvider>
  );
}
