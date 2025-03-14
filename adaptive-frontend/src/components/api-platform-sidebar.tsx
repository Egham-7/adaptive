import { useState } from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar";
import { BarChart3, Key, Settings, AlertCircle, Home } from "lucide-react";

import { UserButton } from "@clerk/clerk-react";
import CommonSidebarHeader from "./sidebar-header";

export default function ApiPlatformSidebar() {
  const [activeItem, setActiveItem] = useState("overview");
  return (
    <Sidebar>
      <CommonSidebarHeader />

      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={activeItem === "overview"}
              onClick={() => setActiveItem("overview")}
              tooltip="Dashboard"
            >
              <Home />
              <span>Dashboard</span>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={activeItem === "usage"}
              onClick={() => setActiveItem("usage")}
              tooltip="Usage Analytics"
            >
              <BarChart3 />
              <span>Usage Analytics</span>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={activeItem === "api-keys"}
              onClick={() => setActiveItem("api-keys")}
              tooltip="API Keys"
            >
              <Key />
              <span>API Keys</span>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={activeItem === "settings"}
              onClick={() => setActiveItem("settings")}
              tooltip="Settings"
            >
              <Settings />
              <span>Settings</span>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={activeItem === "alerts"}
              onClick={() => setActiveItem("alerts")}
              tooltip="Alerts"
            >
              <AlertCircle />
              <span>Alerts</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarContent>

      <SidebarFooter className="p-4">
        <UserButton />
      </SidebarFooter>
    </Sidebar>
  );
}
