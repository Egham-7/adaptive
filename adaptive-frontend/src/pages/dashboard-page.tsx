import { useState } from "react";
import { DashboardHeader } from "@/components/api_platform/dashboard/dashboard-header";
import { UsageOverview } from "@/components/api_platform/dashboard/usage-overview";
import { UsageLimits } from "@/components/api_platform/dashboard/usage-limits";
import { ApiKeyManagement } from "@/components/api_platform/dashboard/api-key-management";

export function DashboardPage() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <>
      <DashboardHeader activeTab={activeTab} setActiveTab={setActiveTab} />

      <div className="p-6">
        {activeTab === "overview" && <UsageOverview />}
        {activeTab === "api-keys" && <ApiKeyManagement />}
        {activeTab === "limits" && <UsageLimits />}
      </div>
    </>
  );
}
