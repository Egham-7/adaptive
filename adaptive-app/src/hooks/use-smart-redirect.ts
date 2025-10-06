"use client";

import { useEffect, useState } from "react";
import { api } from "@/trpc/react";

export function useSmartRedirect() {
	const [redirectPath, setRedirectPath] = useState<string | null>(null);
	const { data: mostRecentProject, isLoading } =
		api.projects.getMostRecent.useQuery();

	useEffect(() => {
		if (isLoading) return;

		if (mostRecentProject) {
			const path = `/api-platform/organizations/${mostRecentProject.organizationId}/projects/${mostRecentProject.id}`;
			setRedirectPath(path);
		} else {
			const path = "/api-platform/organizations";
			setRedirectPath(path);
		}
	}, [mostRecentProject, isLoading]);

	return redirectPath;
}
