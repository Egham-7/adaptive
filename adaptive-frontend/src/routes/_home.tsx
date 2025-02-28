import { HomeLayout } from "@/layouts/home-layout";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/_home")({
  component: HomeLayout,
});
