import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import {
  SignInButton,
  SignUpButton,
  SignedIn,
  SignedOut,
} from "@clerk/clerk-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDown } from "lucide-react";

const navLinks = [
  { href: "#features", label: "Features" },
  { href: "#testimonials", label: "Testimonials" },
  { href: "#pricing", label: "Pricing" },
];

export default function Header() {
  const [redirectPath, setRedirectPath] = useState("/home"); // Default redirect path

  // Update redirect path when user selects a destination
  const handleRedirectPathChange = (path: string) => {
    setRedirectPath(path);
  };

  return (
    <header className="sticky p-6 top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
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
          <SignedIn>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  size="sm"
                  className="bg-gradient-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle flex items-center gap-1"
                >
                  Go to App <ChevronDown className="h-4 w-4 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem asChild>
                  <Link to="/home" className="cursor-pointer flex w-full">
                    Chatbot App
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link
                    to="/api_platform"
                    className="cursor-pointer flex w-full"
                  >
                    API Platform
                  </Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SignedIn>
          <SignedOut>
            {/* Dropdown for destination selection */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="font-medium hover:text-primary-600 hover:bg-primary-50/50 flex items-center gap-1"
                >
                  Destination <ChevronDown className="h-4 w-4 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() => handleRedirectPathChange("/home")}
                >
                  Chatbot App
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleRedirectPathChange("/api-platform")}
                >
                  API Platform
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Sign in button with dynamic redirect */}
            <SignInButton signUpForceRedirectUrl={redirectPath}>
              <Button
                variant="ghost"
                size="sm"
                className="font-medium hover:text-primary-600 hover:bg-primary-50/50"
              >
                Log in
              </Button>
            </SignInButton>

            {/* Sign up button with dynamic redirect */}
            <SignUpButton signInForceRedirectUrl={redirectPath}>
              <Button
                size="sm"
                className="bg-gradient-to-r from-primary-600 to-secondary-600 text-white hover:opacity-90 transition-opacity font-medium shadow-subtle"
              >
                Get Started
              </Button>
            </SignUpButton>
          </SignedOut>
        </div>
      </div>
    </header>
  );
}
