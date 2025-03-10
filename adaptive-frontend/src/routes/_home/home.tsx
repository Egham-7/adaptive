import { createFileRoute } from "@tanstack/react-router";
import Home from "@/pages/home-page";

export const Route = createFileRoute("/_home/home")({
  component: Home,
});
