import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface DashboardHeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export function DashboardHeader({
  activeTab,
  setActiveTab,
}: DashboardHeaderProps) {
  return (
    <div className="flex flex-col space-y-4 p-6 pt-4">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
        <div className="flex items-center gap-2">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="api-keys">API Keys</TabsTrigger>
              <TabsTrigger value="limits">Usage Limits</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
