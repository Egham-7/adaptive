"use client";

import React, { useState } from "react";
import {
  motion,
  AnimatePresence,
  useScroll,
  useMotionValueEvent,
} from "framer-motion";
import { SignedIn, SignedOut, SignInButton, SignUpButton } from "@clerk/nextjs";
import { ChevronDown, Home, Star, FileText, DollarSign, Users, HelpCircle } from "lucide-react";
import Link, { useLinkStatus } from "next/link";
import { HiOutlineDocumentText } from "react-icons/hi2";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { GitHubStarsButton } from "@/components/animate-ui/buttons/github-stars";
import { ModeToggle } from "@/app/_components/mode-toggle";

function LoadingLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  const { pending } = useLinkStatus();

  return (
    <Link href={href} className="flex items-center gap-2">
      {pending && <span className="animate-spin">⟳</span>}
      {children}
    </Link>
  );
}

const navItems = [
  { name: "Features", link: "/features", icon: <Star className="h-4 w-4" /> },
  { name: "Solution", link: "/solution", icon: <FileText className="h-4 w-4" /> },
  { name: "Pricing", link: "/pricing", icon: <DollarSign className="h-4 w-4" /> },
  { name: "About", link: "/about", icon: <Users className="h-4 w-4" /> },
  { name: "Support", link: "/support", icon: <HelpCircle className="h-4 w-4" /> },
];

export const FloatingNav = ({
  className,
}: {
  className?: string;
}) => {
  const { scrollYProgress } = useScroll();
  const [visible, setVisible] = useState(false);

  useMotionValueEvent(scrollYProgress, "change", (current) => {
    if (typeof current === "number") {
      const currentProgress = scrollYProgress.get();
      
      // Simple: show floating nav when scrolled down past the header
      if (currentProgress > 0.08) {
        setVisible(true);
      } else {
        setVisible(false);
      }
    }
  });

  return (
    <AnimatePresence mode="wait">
      <motion.div
        initial={{
          opacity: 1,
          y: -100,
        }}
        animate={{
          y: visible ? 0 : -100,
          opacity: visible ? 1 : 0,
        }}
        transition={{
          duration: 0.2,
        }}
        className={cn(
          "flex max-w-fit fixed top-6 inset-x-0 mx-auto border border-border/40 rounded-full bg-background/80 backdrop-blur-md shadow-lg z-[5000] px-6 py-3 items-center justify-center gap-6",
          className
        )}
      >
        {/* Navigation Items */}
        {navItems.map((navItem, idx: number) => (
          <Link
            key={`link=${idx}`}
            href={navItem.link}
            className="relative text-muted-foreground hover:text-foreground transition-colors duration-200 flex items-center gap-1"
          >
            <span className="block sm:hidden">{navItem.icon}</span>
            <span className="hidden sm:block text-sm font-medium">{navItem.name}</span>
          </Link>
        ))}

        {/* Separator */}
        <div className="h-4 w-px bg-border/60" />

        {/* External Links */}
        <div className="flex items-center gap-3">
          <a
            href="https://docs.llmadaptive.uk/"
            target="_blank"
            rel="noopener noreferrer" 
            className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors duration-200"
          >
            <HiOutlineDocumentText className="h-4 w-4" />
            <span className="hidden sm:block text-sm">Docs</span>
          </a>
          
          <GitHubStarsButton 
            username="Egham-7" 
            repo="adaptive" 
            formatted={true}
            className="text-xs h-8 scale-90"
          />
        </div>

        {/* Separator */}
        <div className="h-4 w-px bg-border/60" />

        {/* Auth Buttons */}
        <div className="flex items-center gap-2">
          <SignedOut>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-sm font-medium text-muted-foreground hover:text-foreground"
                >
                  Login
                  <ChevronDown className="h-3 w-3 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem asChild>
                  <SignInButton signUpForceRedirectUrl="/chat-platform">
                    <Button variant="ghost" className="w-full justify-start">
                      Chatbot App
                    </Button>
                  </SignInButton>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <SignInButton signUpForceRedirectUrl="/api-platform/organizations">
                    <Button variant="ghost" className="w-full justify-start">
                      API Platform
                    </Button>
                  </SignInButton>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  size="sm"
                  className="text-sm font-medium relative border border-border/40 bg-primary text-primary-foreground hover:bg-primary/90 rounded-full px-4 py-2"
                >
                  Get Started
                  <ChevronDown className="h-3 w-3 ml-1" />
                  <span className="absolute inset-x-0 w-1/2 mx-auto -bottom-px bg-gradient-to-r from-transparent via-primary/60 to-transparent h-px" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem asChild>
                  <SignUpButton signInForceRedirectUrl="/chat-platform">
                    <Button variant="ghost" className="w-full justify-start">
                      Chatbot App
                    </Button>
                  </SignUpButton>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <SignUpButton signInForceRedirectUrl="/api-platform/organizations">
                    <Button variant="ghost" className="w-full justify-start">
                      API Platform
                    </Button>
                  </SignUpButton>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SignedOut>

          <SignedIn>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-sm font-medium text-muted-foreground hover:text-foreground"
                >
                  My Account
                  <ChevronDown className="h-3 w-3 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem asChild>
                  <LoadingLink href="/chat-platform">
                    <Button variant="ghost" className="w-full justify-start">
                      Chatbot App
                    </Button>
                  </LoadingLink>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <LoadingLink href="/api-platform/organizations">
                    <Button variant="ghost" className="w-full justify-start">
                      API Platform
                    </Button>
                  </LoadingLink>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SignedIn>

          <ModeToggle />
        </div>
      </motion.div>
    </AnimatePresence>
  );
};