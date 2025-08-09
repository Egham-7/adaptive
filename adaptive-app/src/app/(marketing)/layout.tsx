import { Suspense } from "react";
import FloatingHeader from "@/app/_components/landing_page/floating-header";
import Header from "@/app/_components/landing_page/header";

export default function MarketingLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<Suspense>
			<Header />
			<FloatingHeader />
			{children}
		</Suspense>
	);
}
