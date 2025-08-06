import { HiOutlineDocumentText } from "react-icons/hi2";

export type MenuItem = {
	name: string;
	href: string;
	external?: boolean;
};

export type IconMenuItem = {
	name: string;
	href: string;
	icon: React.ComponentType<{ size?: number; "aria-hidden"?: boolean }>;
	external: boolean;
};

export const menuItems: MenuItem[] = [
	{ name: "Features", href: "/features" },
	{ name: "Solution", href: "/solution" },
	{ name: "Pricing", href: "/pricing" },
	{ name: "About", href: "/about" },
	{ name: "Support", href: "/support" },
];

export const iconMenuItems: IconMenuItem[] = [
	{
		name: "Docs",
		href: "https://docs.llmadaptive.uk/",
		icon: HiOutlineDocumentText,
		external: true,
	},
];
