"use client";

import { useEffect, useState } from "react";

const LAST_PROJECT_KEY = "adaptive_last_project_id";
const LAST_ORG_KEY = "adaptive_last_org_id";

export function useSmartRedirect() {
	const [redirectPath, setRedirectPath] = useState<string | null>(null);

	useEffect(() => {
		const lastProjectId = localStorage.getItem(LAST_PROJECT_KEY);
		const lastOrgId = localStorage.getItem(LAST_ORG_KEY);

		if (lastProjectId && lastOrgId) {
			const path = `/api-platform/organizations/${lastOrgId}/projects/${lastProjectId}`;
			setRedirectPath(path);
		} else if (lastOrgId) {
			const path = `/api-platform/organizations/${lastOrgId}`;
			setRedirectPath(path);
		} else {
			const path = "/api-platform/organizations";
			setRedirectPath(path);
		}
	}, []);

	return redirectPath;
}

export function setLastProject(orgId: string, projectId: string) {
	if (typeof window !== "undefined") {
		localStorage.setItem(LAST_ORG_KEY, orgId);
		localStorage.setItem(LAST_PROJECT_KEY, projectId);
	}
}

export function setLastOrganization(orgId: string) {
	if (typeof window !== "undefined") {
		localStorage.setItem(LAST_ORG_KEY, orgId);
		localStorage.removeItem(LAST_PROJECT_KEY);
	}
}

export function clearLastProject() {
	if (typeof window !== "undefined") {
		localStorage.removeItem(LAST_PROJECT_KEY);
		localStorage.removeItem(LAST_ORG_KEY);
	}
}

export function clearLastOrganization() {
	if (typeof window !== "undefined") {
		localStorage.removeItem(LAST_ORG_KEY);
		localStorage.removeItem(LAST_PROJECT_KEY);
	}
}
