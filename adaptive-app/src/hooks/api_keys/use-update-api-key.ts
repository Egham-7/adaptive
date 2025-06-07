import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useUpdateApiKey = () => {
  const utils = api.useUtils();

  return api.api_keys.update.useMutation({
    onMutate: async (newData) => {
      // Cancel outgoing refetches
      await utils.api_keys.list.cancel();
      await utils.api_keys.getById.cancel({ id: newData.id });

      // Snapshot previous values
      const previousList = utils.api_keys.list.getData();
      const previousItem = utils.api_keys.getById.getData({ id: newData.id });

      // Optimistically update the cache
      if (previousList) {
        utils.api_keys.list.setData(undefined, (old) =>
          old?.map((item) =>
            item.id === newData.id ? { ...item, ...newData } : item,
          ),
        );
      }

      if (previousItem) {
        utils.api_keys.getById.setData({ id: newData.id }, (old) =>
          old ? { ...old, ...newData } : old,
        );
      }

      return { previousList, previousItem };
    },
    onError: (error, newData, context) => {
      // Rollback on error
      if (context?.previousList) {
        utils.api_keys.list.setData(undefined, context.previousList);
      }
      if (context?.previousItem) {
        utils.api_keys.getById.setData(
          { id: newData.id },
          context.previousItem,
        );
      }
      toast.error(error.message || "Failed to update API key");
    },
    onSettled: () => {
      // Always refetch after error or success
      utils.api_keys.list.invalidate();
    },
    onSuccess: () => {
      toast.success("API key updated successfully!");
    },
  });
};
