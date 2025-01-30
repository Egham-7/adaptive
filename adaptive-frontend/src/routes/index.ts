import { rootRoute } from "./root";
import Home from "@/pages/home";
import { createRouter, createRoute } from "@tanstack/react-router";

export const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Home,
});

// Build the route tree
export const routeTree = rootRoute.addChildren([indexRoute]);

// Create the router instance
export const router = createRouter({
  routeTree,
});

// Register router for type safety
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
