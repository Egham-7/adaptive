import "../styles/globals.css";
import { SidebarProvider } from "@/components/ui/sidebar";
import { ThemeProvider } from "@/context/theme-provider";
import { TRPCReactProvider } from "@/trpc/react";
import { HydrateClient } from "@/trpc/server";
import { ClerkProvider } from "@clerk/nextjs";
import { Analytics } from "@vercel/analytics/next";
import type { Metadata } from "next";
import { Lora, Outfit, Roboto_Mono } from "next/font/google";
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
		<ClerkProvider
			appearance={{
				variables: {
					borderRadius: "var(--radius)",
					fontFamily: "var(--font-sans)",
				},
				elements: {
					// Use Tailwind classes directly - they automatically adapt to theme
					formButtonPrimary:
						"bg-primary text-primary-foreground hover:bg-primary/90",
					card: "bg-card text-card-foreground border border-border",
					headerTitle: "text-foreground",
					headerSubtitle: "text-muted-foreground",
					socialButtonsBlockButton:
						"border border-border text-foreground bg-background hover:bg-muted",
					formFieldInput:
						"border border-border bg-background text-foreground focus:ring-2 focus:ring-ring",
					footerActionLink: "text-primary hover:text-primary/80",
					formFieldLabel: "text-foreground",
					formFieldAction: "text-primary hover:text-primary/80",
					identityPreviewText: "text-foreground",
					identityPreviewEditButton: "text-primary hover:text-primary/80",
					accordionTriggerButton: "text-foreground hover:text-muted-foreground",
					accordionContent: "text-muted-foreground",
					formResendCodeLink: "text-primary hover:text-primary/80",
					otpCodeFieldInput:
						"border border-border bg-background text-foreground focus:ring-2 focus:ring-ring",
					formFieldSuccessText: "text-green-600",
					formFieldErrorText: "text-destructive",
					alert: "border border-border bg-card text-card-foreground",
					alertText: "text-foreground",
					rootBox: "bg-background",
					navbar: "bg-card border-b border-border",
					navbarMobileMenuButton: "text-foreground",
				},
			}}
		>
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
					<Analytics />
					<Toaster />
				</body>
			</html>
		</ClerkProvider>
	);
}
