import { Outlet } from "@tanstack/react-router";
import { ThemeProvider } from "@/context/theme-provider";
export function RootLayout() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <div className="flex min-h-screen flex-col">
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </ThemeProvider>
  );
}
