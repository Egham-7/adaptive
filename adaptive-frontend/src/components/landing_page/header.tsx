import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { SignInButton, SignUpButton } from "@clerk/clerk-react";

const navLinks = [
  { href: "#features", label: "Features" },
  { href: "#testimonials", label: "Testimonials" },
  { href: "#pricing", label: "Pricing" },
];

export default function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex gap-8 items-center">
          <Link to="/" className="flex items-center">
            <span className="font-display text-xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              Adaptive
            </span>
          </Link>
          <nav className="hidden md:flex gap-8">
            {navLinks.map((link, index) => (
              <Link
                key={index}
                to={link.href}
                className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary-600 hover:scale-105 transform duration-200"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-4">
          <SignInButton>
            <Button
              variant="ghost"
              size="sm"
              className="font-medium hover:text-primary-600 hover:bg-primary-50/50"
            >
              Log in
            </Button>
          </SignInButton>
          <SignUpButton>
            <Button
              size="sm"
              className="bg-gradient-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle"
            >
              Get Started
            </Button>
          </SignUpButton>
        </div>
      </div>
    </header>
  );
}
