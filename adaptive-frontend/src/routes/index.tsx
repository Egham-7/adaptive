import LandingPage from "@/pages/landing_page";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: LandingPage,
});
