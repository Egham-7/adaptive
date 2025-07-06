import type { RouterInputs, RouterOutputs } from "./index";

// ---- Output Types ----

/**
 * Organization member type
 */
export type OrganizationMember = {
	id: string;
	userId: string;
	organizationId: string;
	role: string;
	createdAt: Date;
	updatedAt: Date;
};

/**
 * The type for a single organization item in the list.
 * Includes all organization details like name, description, members, etc.
 */
export type OrganizationListItem =
	RouterOutputs["organizations"]["getAll"][number];

/**
 * The type for a single, fully-detailed organization.
 * Same as list item but explicitly typed for single item operations.
 */
export type OrganizationDetails = RouterOutputs["organizations"]["getById"];

/**
 * The type for the response when creating a new organization.
 * Returns the created organization details.
 */
export type OrganizationCreateResponse =
	RouterOutputs["organizations"]["create"];

/**
 * The type for the response when updating an organization.
 * Returns the updated organization details.
 */
export type OrganizationUpdateResponse =
	RouterOutputs["organizations"]["update"];

/**
 * The type for the response when deleting an organization.
 * Simple success confirmation.
 */
export type OrganizationDeleteResponse =
	RouterOutputs["organizations"]["delete"];

/**
 * The type for the complete list of organizations.
 */
export type OrganizationsList = RouterOutputs["organizations"]["getAll"];

// ---- Input Types ----

/**
 * The type for the input when creating a new organization.
 */
export type OrganizationCreateInput = RouterInputs["organizations"]["create"];

/**
 * The type for the input when updating an organization.
 */
export type OrganizationUpdateInput = RouterInputs["organizations"]["update"];

/**
 * The type for the input when getting an organization by ID.
 */
export type OrganizationGetByIdInput = RouterInputs["organizations"]["getById"];

/**
 * The type for the input when deleting an organization.
 */
export type OrganizationDeleteInput = RouterInputs["organizations"]["delete"];

// ---- Utility Types ----

/**
 * The core organization data without metadata like created_at, updated_at.
 */
export type OrganizationCore = Pick<
	OrganizationListItem,
	"id" | "name" | "description" | "ownerId"
>;

/**
 * Organization data for forms (without server-generated fields).
 */
export type OrganizationFormData = Pick<
	OrganizationCreateInput,
	"name" | "description"
>;

/**
 * Organization data for updates (without immutable fields).
 */
export type OrganizationUpdateData = Omit<OrganizationUpdateInput, "id">;

/**
 * Organization member role types.
 */
export type OrganizationMemberRole = OrganizationMember["role"];
