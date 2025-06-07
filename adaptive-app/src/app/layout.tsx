import "../styles/globals.css";
import { SidebarProvider } from "@/components/ui/sidebar";
import { ThemeProvider } from "@/context/theme-provider";
import { TRPCReactProvider } from "@/trpc/react";
import { HydrateClient } from "@/trpc/server";
import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata } from "next";
import { Lora, Outfit, Roboto_Mono } from "next/font/google"; // Import all three fonts
import { Toaster } from "sonner";

// Configure Outfit for sans-serif
const outfit = Outfit({
	subsets: ["latin"],
	variable: "--font-sans",
});

// Configure Roboto Mono for monospace
const robotoMono = Roboto_Mono({
	subsets: ["latin"],
	variable: "--font-mono",
});

// Configure Lora for serif
const lora = Lora({
	subsets: ["latin"],
	variable: "--font-serif",
});

export const metadata: Metadata = {
	title: "Adaptive",
	description: "Adaptive",
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<ClerkProvider>
			<html lang="en" suppressHydrationWarning>
				{/* Apply all font variables to the body */}
				<body
					className={`${outfit.variable} ${robotoMono.variable} ${lora.variable} bg-background`}
				>
					<ThemeProvider
						attribute="class"
						defaultTheme="system"
						enableSystem
						disableTransitionOnChange
					>
						<TRPCReactProvider>
							<HydrateClient>
								<SidebarProvider>
									<div className="min-h-screen w-full bg-background">
										{children}
									</div>
								</SidebarProvider>
							</HydrateClient>
						</TRPCReactProvider>
					</ThemeProvider>
					<Toaster />
				</body>
			</html>
		</ClerkProvider>
	);
}
