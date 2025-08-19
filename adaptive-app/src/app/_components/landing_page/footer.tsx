import Link from "next/link";
import { Logo } from "../logo";

const links = [
	{
		group: "Product",
		items: [
			{ title: "Features", href: "/features" },
			{ title: "Solution", href: "/solution" },
			{ title: "Pricing", href: "/pricing" },
		],
	},
	{
		group: "Resources",
		items: [
			{ title: "Documentation", href: "https://docs.llmadaptive.uk/" },
			{ title: "Support", href: "/support" },
			{ title: "GitHub", href: "https://github.com/Egham-7/adaptive" },
		],
	},
	{
		group: "Company",
		items: [
			{ title: "About", href: "/about" },
			{ title: "Contact", href: "/contact" },
		],
	},
	{
		group: "Legal",
		items: [
			{ title: "Terms of Service", href: "/terms-of-service" },
			{ title: "Privacy Policy", href: "/privacy-policy" },
		],
	},
];

export default function FooterSection() {
	return (
		<footer
			id="about"
			className="border-b bg-background pt-20 dark:bg-transparent"
		>
			<div className="mx-auto max-w-5xl px-6">
				<div className="grid gap-12 md:grid-cols-5">
					<div className="md:col-span-2">
						<Link href="/" aria-label="go home" className="block size-fit">
							<Logo />
						</Link>
						<p className="mt-4 max-w-xs text-muted-foreground text-sm">
							Optimize performance and cut costs with Adaptive's LLM-driven
							infrastructure for all your AI workloads.
						</p>
					</div>
					<nav
						className="grid grid-cols-2 gap-6 md:col-span-3 md:grid-cols-4"
						aria-label="Footer navigation"
					>
						{links.map((link) => (
							<div key={link.group} className="space-y-4 text-sm">
								<h3 className="block font-medium">{link.group}</h3>
								<ul className="space-y-2">
									{link.items.map((item) => (
										<li key={item.title}>
											<Link
												href={item.href}
												className="block text-muted-foreground duration-150 hover:text-primary"
											>
												<span>{item.title}</span>
											</Link>
										</li>
									))}
								</ul>
							</div>
						))}
					</nav>
				</div>
				<div className="mt-12 flex flex-wrap items-end justify-between gap-6 border-t py-6">
					<span className="order-last block text-center text-muted-foreground text-sm md:order-first">
						© {new Date().getFullYear()} Botir Khaltaev, Kendrick Lwin, Mohamed
						El Amine Atoui. All rights reserved
					</span>
					<nav
						className="order-first flex flex-wrap justify-center gap-6 text-sm md:order-last"
						aria-label="Social media links"
					>
						<ul className="flex gap-6">
							<li>
								<a
									href="https://x.com/Adaptivellm"
									target="_blank"
									rel="noopener noreferrer"
									aria-label="Follow us on X (formerly Twitter)"
									className="block text-muted-foreground hover:text-primary"
								>
									<svg
										className="size-6"
										xmlns="http://www.w3.org/2000/svg"
										width="1em"
										height="1em"
										viewBox="0 0 24 24"
									>
										<title>X (formerly Twitter) logo</title>
										<path
											fill="currentColor"
											d="M10.488 14.651L15.25 21h7l-7.858-10.478L20.93 3h-2.65l-5.117 5.886L8.75 3h-7l7.51 10.015L2.32 21h2.65zM16.25 19L5.75 5h2l10.5 14z"
										/>
									</svg>
								</a>
							</li>
							<li>
								<a
									href="https://linkedin.com/company/llmadaptive"
									target="_blank"
									rel="noopener noreferrer"
									aria-label="Connect with us on LinkedIn"
									className="block text-muted-foreground hover:text-primary"
								>
									<svg
										className="size-6"
										xmlns="http://www.w3.org/2000/svg"
										width="1em"
										height="1em"
										viewBox="0 0 24 24"
									>
										<title>LinkedIn logo</title>
										<path
											fill="currentColor"
											d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93zM6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37z"
										/>
									</svg>
								</a>
							</li>
							<li>
								<a
									href="https://github.com/Egham-7/adaptive"
									target="_blank"
									rel="noopener noreferrer"
									aria-label="View our code on GitHub"
									className="block text-muted-foreground hover:text-primary"
								>
									<svg
										className="size-6"
										xmlns="http://www.w3.org/2000/svg"
										width="1em"
										height="1em"
										viewBox="0 0 24 24"
									>
										<title>GitHub logo</title>
										<path
											fill="currentColor"
											d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5c.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34c-.46-1.16-1.11-1.47-1.11-1.47c-.91-.62.07-.6.07-.6c1 .07 1.53 1.03 1.53 1.03c.87 1.52 2.34 1.07 2.91.83c.09-.65.35-1.09.63-1.34c-2.22-.25-4.55-1.11-4.55-4.92c0-1.11.38-2 1.03-2.71c-.1-.25-.45-1.29.1-2.64c0 0 .84-.27 2.75 1.02c.79-.22 1.65-.33 2.5-.33c.85 0 1.71.11 2.5.33c1.91-1.29 2.75-1.02 2.75-1.02c.55 1.35.2 2.39.1 2.64c.65.71 1.03 1.6 1.03 2.71c0 3.82-2.34 4.66-4.57 4.91c.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"
										/>
									</svg>
								</a>
							</li>
						</ul>
					</nav>
				</div>
			</div>
		</footer>
	);
}
