"use client";

import { cn } from "@/lib/utils";
import Link, { type LinkProps } from "next/link";
import React, { useState, createContext, useContext } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import { Slot } from "@radix-ui/react-slot";

interface Links {
  label: string;
  href: string;
  icon: React.JSX.Element | React.ReactNode;
}

interface SidebarContextProps {
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  animate: boolean;
}

const SidebarContext = createContext<SidebarContextProps | undefined>(
  undefined
);

export const useSidebar = () => {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider");
  }
  return context;
};

export const SidebarProvider = ({
  children,
  open: openProp,
  setOpen: setOpenProp,
  animate = true,
}: {
  children: React.ReactNode;
  open?: boolean;
  setOpen?: React.Dispatch<React.SetStateAction<boolean>>;
  animate?: boolean;
}) => {
  const [openState, setOpenState] = useState(false);

  const open = openProp !== undefined ? openProp : openState;
  const setOpen = setOpenProp !== undefined ? setOpenProp : setOpenState;

  return (
    <SidebarContext.Provider value={{ open, setOpen, animate }}>
      {children}
    </SidebarContext.Provider>
  );
};

export const Sidebar = ({
  children,
  open,
  setOpen,
  animate,
  className,
  ...props
}: {
  children: React.ReactNode;
  open?: boolean;
  setOpen?: React.Dispatch<React.SetStateAction<boolean>>;
  animate?: boolean;
  className?: string;
} & React.ComponentProps<"div">) => {
  return (
    <SidebarProvider open={open} setOpen={setOpen} animate={animate}>
      <div className={className} {...props}>
        {children}
      </div>
    </SidebarProvider>
  );
};

export const SidebarBody = (props: React.ComponentProps<typeof motion.div>) => {
  return (
    <>
      <DesktopSidebar {...props} />
      <MobileSidebar {...(props as React.ComponentProps<"div">)} />
    </>
  );
};

export const DesktopSidebar = ({
  className,
  children,
  ...props
}: React.ComponentProps<typeof motion.div>) => {
  const { open, setOpen, animate } = useSidebar();
  return (
    <motion.div
      className={cn(
        "h-full px-4 py-4 hidden md:flex md:flex-col bg-neutral-100 dark:bg-neutral-800 w-[300px] flex-shrink-0",
        className
      )}
      animate={{
        width: animate ? (open ? "300px" : "60px") : "300px",
      }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export const MobileSidebar = ({
  className,
  children,
  ...props
}: React.ComponentProps<"div">) => {
  const { open, setOpen } = useSidebar();
  return (
    <>
      <div
        className={cn(
          "h-10 px-4 py-4 flex flex-row md:hidden items-center justify-between bg-neutral-100 dark:bg-neutral-800 w-full"
        )}
        {...props}
      >
        <div className="flex justify-end z-20 w-full">
          <Menu
            className="text-neutral-800 dark:text-neutral-200 cursor-pointer"
            onClick={() => setOpen(!open)}
          />
        </div>
        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ x: "-100%", opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: "-100%", opacity: 0 }}
              transition={{
                duration: 0.3,
                ease: "easeInOut",
              }}
              className={cn(
                "fixed h-full w-full inset-0 bg-white dark:bg-neutral-900 p-10 z-[100] flex flex-col justify-between",
                className
              )}
            >
              <div
                className="absolute right-10 top-10 z-50 text-neutral-800 dark:text-neutral-200 cursor-pointer"
                onClick={() => setOpen(!open)}
              >
                <X />
              </div>
              {children}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
};

export const SidebarLink = ({
  link,
  className,
  ...props
}: {
  link: Links;
  className?: string;
  props?: LinkProps;
}) => {
  const { open, animate } = useSidebar();
  return (
    <Link
      href={link.href}
      className={cn(
        "flex items-center justify-start gap-2 group/sidebar py-2",
        className
      )}
      {...props}
    >
      {link.icon}
      <motion.span
        animate={{
          display: animate ? (open ? "inline-block" : "none") : "inline-block",
          opacity: animate ? (open ? 1 : 0) : 1,
        }}
        className="text-neutral-700 dark:text-neutral-200 text-sm group-hover/sidebar:translate-x-1 transition duration-150 whitespace-pre inline-block !p-0 !m-0"
      >
        {link.label}
      </motion.span>
    </Link>
  );
};

// Additional sidebar components for compatibility
export const SidebarContent = ({ children, className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("flex flex-col flex-1", className)} {...props}>
    {children}
  </div>
);

export const SidebarRail = ({ className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("w-px bg-border absolute inset-y-0 left-0", className)} {...props} />
);

export const SidebarSeparator = ({ className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("w-full h-px bg-border my-2", className)} {...props} />
);

export const SidebarTrigger = ({ children, className, ...props }: React.ComponentProps<"button">) => (
  <button className={cn("p-2 hover:bg-accent rounded", className)} {...props}>
    {children}
  </button>
);

export const SidebarGroup = ({ children, className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("py-2", className)} {...props}>
    {children}
  </div>
);

export const SidebarGroupContent = ({ children, className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("pl-2", className)} {...props}>
    {children}
  </div>
);

export const SidebarGroupLabel = ({ children, className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("text-sm font-medium text-muted-foreground mb-1", className)} {...props}>
    {children}
  </div>
);

export const SidebarMenu = ({ children, className, ...props }: React.ComponentProps<"ul">) => (
  <ul className={cn("space-y-1", className)} {...props}>
    {children}
  </ul>
);

export const SidebarMenuButton = ({ 
  children, 
  className, 
  asChild = false, 
  ...props 
}: React.ComponentProps<"button"> & { asChild?: boolean }) => {
  const Comp = asChild ? Slot : "button";
  
  return (
    <Comp className={cn("w-full text-left p-2 hover:bg-accent rounded flex items-center gap-2", className)} {...props}>
      {children}
    </Comp>
  );
};

export const SidebarMenuItem = ({ children, className, ...props }: React.ComponentProps<"li">) => (
  <li className={cn("", className)} {...props}>
    {children}
  </li>
);

export const SidebarFooter = ({ children, className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("mt-auto pt-4", className)} {...props}>
    {children}
  </div>
);

export const SidebarHeader = ({ children, className, ...props }: React.ComponentProps<"div">) => (
  <div className={cn("pb-4", className)} {...props}>
    {children}
  </div>
);
