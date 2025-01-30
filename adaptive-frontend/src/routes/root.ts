import { createRootRoute } from "@tanstack/react-router";
import { RootLayout } from "@/layouts/root-layout";

export const rootRoute = createRootRoute({
  component: RootLayout,
});
