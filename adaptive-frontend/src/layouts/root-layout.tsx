import { Outlet } from "@tanstack/react-router";
import Header from "@/components/landing_page/header";
import Footer from "@/components/landing_page/footer";

export function RootLayout() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="flex-1">
        <Outlet />
      </main>

      <Footer />
    </div>
  );
}
