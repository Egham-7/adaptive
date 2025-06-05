// src/components/chat/chat-sidebar/SidebarNavFooter.tsx

import { UserButton } from "@clerk/nextjs";
import {
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@/components/ui/sidebar";
import { ModeToggle } from "../../mode-toggle";

export function SidebarNavFooter() {
  return (
    <SidebarFooter>
      <SidebarSeparator />
      <SidebarMenu className="flex-row items-center justify-between p-2">
        <SidebarMenuItem>
          <SidebarMenuButton asChild>
            <UserButton
              appearance={{
                elements: {
                  userButtonAvatarBox: "w-8 h-8",
                },
              }}
            />
          </SidebarMenuButton>
        </SidebarMenuItem>
        <SidebarMenuItem>
          <SidebarMenuButton asChild>
            <ModeToggle />
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarFooter>
  );
}
