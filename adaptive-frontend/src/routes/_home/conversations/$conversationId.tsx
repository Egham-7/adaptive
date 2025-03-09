import { createFileRoute } from "@tanstack/react-router";
import ConversationPage from "@/pages/chat";

export const Route = createFileRoute("/_home/conversations/$conversationId")({
  component: ConversationPage,
});
