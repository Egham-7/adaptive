import { SidebarHeader } from "./ui/sidebar";
import { Layers } from "lucide-react";

export default function CommonSidebarHeader() {
  return (
    <SidebarHeader className="flex items-center px-4 py-2">
      <div className="flex items-center gap-2">
        <Layers className="h-6 w-6 text-primary" />
        <span className="text-xl font-bold">Adaptive</span>
      </div>
    </SidebarHeader>
  );
}
