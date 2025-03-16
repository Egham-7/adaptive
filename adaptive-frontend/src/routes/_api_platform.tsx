import { APIPlatformLayout } from "@/layouts/api-platform-layout";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/_api_platform")({
  component: APIPlatformLayout,
});
