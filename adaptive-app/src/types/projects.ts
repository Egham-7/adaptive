import type { RouterInputs, RouterOutputs } from "./index";

// ---- Output Types ----

/**
 * Project member type
 */
export type ProjectMember = {
	id: string;
	userId: string;
	projectId: string;
	role: string;
	createdAt: Date;
	updatedAt: Date;
};

/**
 * The type for a single project item in the list.
 * Includes all project details like name, description, status, etc.
 */
export type ProjectListItem =
	RouterOutputs["projects"]["getByOrganization"][number];

/**
 * The type for a single, fully-detailed project.
 * Same as list item but explicitly typed for single item operations.
 */
export type ProjectDetails = RouterOutputs["projects"]["getById"];

/**
 * The type for the response when creating a new project.
 * Returns the created project details.
 */
export type ProjectCreateResponse = RouterOutputs["projects"]["create"];

/**
 * The type for the response when updating a project.
 * Returns the updated project details.
 */
export type ProjectUpdateResponse = RouterOutputs["projects"]["update"];

/**
 * The type for the response when deleting a project.
 * Simple success confirmation.
 */
export type ProjectDeleteResponse = RouterOutputs["projects"]["delete"];

/**
 * The type for the complete list of projects for an organization.
 */
export type ProjectsList = RouterOutputs["projects"]["getByOrganization"];

// ---- Input Types ----

/**
 * The type for the input when creating a new project.
 */
export type ProjectCreateInput = RouterInputs["projects"]["create"];

/**
 * The type for the input when updating a project.
 */
export type ProjectUpdateInput = RouterInputs["projects"]["update"];

/**
 * The type for the input when getting a project by ID.
 */
export type ProjectGetByIdInput = RouterInputs["projects"]["getById"];

/**
 * The type for the input when deleting a project.
 */
export type ProjectDeleteInput = RouterInputs["projects"]["delete"];

/**
 * The type for the input when getting projects by organization.
 */
export type ProjectsByOrganizationInput =
	RouterInputs["projects"]["getByOrganization"];

// ---- Utility Types ----

/**
 * The status values that a project can have.
 */
export type ProjectStatus = ProjectListItem["status"];

/**
 * The core project data without metadata like created_at, updated_at.
 */
export type ProjectCore = Pick<
	ProjectListItem,
	"id" | "name" | "description" | "status" | "organizationId"
>;

/**
 * Project data for forms (without server-generated fields).
 */
export type ProjectFormData = Pick<
	ProjectCreateInput,
	"name" | "description" | "status" | "organizationId"
>;

/**
 * Project data for updates (without immutable fields).
 */
export type ProjectUpdateData = Omit<ProjectUpdateInput, "id">;

/**
 * Project member role types.
 */
export type ProjectMemberRole = ProjectMember["role"];
