import { api } from "@/trpc/react";

export const useMessageById = (id: string) => {
  return api.messages.getById.useQuery(
    { id },
    {
      enabled: !!id,
    },
  );
};
