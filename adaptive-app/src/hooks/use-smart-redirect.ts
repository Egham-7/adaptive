"use client";

import { useEffect, useState } from "react";

const LAST_PROJECT_KEY = "adaptive_last_project_id";
const LAST_ORG_KEY = "adaptive_last_org_id";

export function useSmartRedirect() {
	const [redirectPath, setRedirectPath] = useState<string | null>(null);

	useEffect(() => {
		const lastProjectId = localStorage.getItem(LAST_PROJECT_KEY);
		const lastOrgId = localStorage.getItem(LAST_ORG_KEY);

		console.log("🔍 useSmartRedirect - Reading from localStorage:", {
			lastProjectId,
			lastOrgId,
		});

		if (lastProjectId && lastOrgId) {
			const path = `/api-platform/organizations/${lastOrgId}/projects/${lastProjectId}`;
			console.log("✅ useSmartRedirect - Redirecting to project:", path);
			setRedirectPath(path);
		} else if (lastOrgId) {
			const path = `/api-platform/organizations/${lastOrgId}`;
			console.log("✅ useSmartRedirect - Redirecting to organization:", path);
			setRedirectPath(path);
		} else {
			const path = "/api-platform/organizations";
			console.log(
				"✅ useSmartRedirect - Redirecting to organizations list:",
				path,
			);
			setRedirectPath(path);
		}
	}, []);

	return redirectPath;
}

export function setLastProject(orgId: string, projectId: string) {
	if (typeof window !== "undefined") {
		console.log("💾 setLastProject - Saving to localStorage:", {
			orgId,
			projectId,
		});
		localStorage.setItem(LAST_ORG_KEY, orgId);
		localStorage.setItem(LAST_PROJECT_KEY, projectId);
		console.log("✅ setLastProject - Saved successfully");
	}
}

export function setLastOrganization(orgId: string) {
	if (typeof window !== "undefined") {
		console.log("💾 setLastOrganization - Saving to localStorage:", { orgId });
		localStorage.setItem(LAST_ORG_KEY, orgId);
		localStorage.removeItem(LAST_PROJECT_KEY);
		console.log("✅ setLastOrganization - Saved successfully, cleared project");
	}
}

export function clearLastProject() {
	if (typeof window !== "undefined") {
		console.log("🗑️ clearLastProject - Clearing localStorage");
		localStorage.removeItem(LAST_PROJECT_KEY);
		localStorage.removeItem(LAST_ORG_KEY);
		console.log("✅ clearLastProject - Cleared successfully");
	}
}
