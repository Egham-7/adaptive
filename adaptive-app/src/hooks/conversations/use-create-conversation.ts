import { api } from "@/trpc/react";
import { useRouter } from "next/navigation";

export const useCreateConversation = () => {
  const utils = api.useUtils();
  const router = useRouter();

  return api.conversations.create.useMutation({
    onSuccess: (newConversation) => {
      // Invalidate the list to refetch and show the new item
      utils.conversations.list.invalidate();
      // Navigate to the newly created chat
      router.push(`/conversations/${newConversation.id}`);
    },
    // You could add an onError callback here to show a toast notification
  });
};
