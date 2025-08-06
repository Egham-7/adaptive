"use client";

import { Menu, X } from "lucide-react";

interface MobileMenuButtonProps {
	menuState: boolean;
	onToggle: () => void;
}

export function MobileMenuButton({
	menuState,
	onToggle,
}: MobileMenuButtonProps) {
	return (
		<button
			type="button"
			onClick={onToggle}
			aria-label={menuState === true ? "Close Menu" : "Open Menu"}
			aria-expanded={menuState}
			className="-m-2.5 -mr-4 relative z-30 block cursor-pointer p-2.5 lg:hidden"
		>
			<Menu className="m-auto size-6 in-data-[state=active]:rotate-180 in-data-[state=active]:scale-0 in-data-[state=active]:opacity-0 duration-200" />
			<X className="-rotate-180 absolute inset-0 m-auto size-6 in-data-[state=active]:rotate-0 in-data-[state=active]:scale-100 scale-0 in-data-[state=active]:opacity-100 opacity-0 duration-200" />
		</button>
	);
}
