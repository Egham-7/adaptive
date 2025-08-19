"use client";

import { useUser } from "@clerk/nextjs";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/trpc/react";

export function useTourCompletion() {
	const { user, isLoaded } = useUser();
	const { data: preferences, isPending } = api.user.getPreferences.useQuery(
		undefined,
		{
			enabled: isLoaded && !!user,
		},
	);

	const [isTourCompleted, setIsTourCompletedState] = useState<boolean>(false);

	// Update local state when preferences are loaded
	useEffect(() => {
		if (preferences?.tourCompleted !== undefined) {
			setIsTourCompletedState(preferences.tourCompleted);
		}
	}, [preferences?.tourCompleted]);

	const updateMetadataMutation = api.user.updateMetadata.useMutation();

	const setIsTourCompleted = useCallback(
		async (completed: boolean) => {
			if (!user) return;

			try {
				setIsTourCompletedState(completed);
				await updateMetadataMutation.mutateAsync({
					tourCompleted: completed,
				});

				// Update the user's metadata in Clerk
				await user.reload();
			} catch (error) {
				console.error("Failed to update tour completion state:", error);
				// Revert on error
				setIsTourCompletedState(!completed);
			}
		},
		[user, updateMetadataMutation],
	);

	return {
		isTourCompleted,
		setIsTourCompleted,
		isLoading: !isLoaded || isPending || updateMetadataMutation.isPending,
	};
}
