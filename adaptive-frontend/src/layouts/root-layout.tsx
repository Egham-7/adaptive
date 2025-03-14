import { Outlet } from "@tanstack/react-router";
import { ThemeProvider } from "@/context/theme-provider";
export function RootLayout() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <main className="flex-1">
        <Outlet />
      </main>
    </ThemeProvider>
  );
}
