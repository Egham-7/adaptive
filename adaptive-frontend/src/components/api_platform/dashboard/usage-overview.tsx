import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ProviderUsageTable } from "./usage-overview/provider-usage-table";
import { CostBreakdown } from "./usage-overview/cost-breakdown";
import { UsageChart } from "./usage-overview/usage-chart";

export function UsageOverview() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Requests
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">45,231</div>
            <p className="text-xs text-muted-foreground">
              +20.1% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tokens</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2.4M</div>
            <p className="text-xs text-muted-foreground">
              +15% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$342.65</div>
            <p className="text-xs text-muted-foreground">+5% from last month</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cost Savings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$128.40</div>
            <p className="text-xs text-muted-foreground">
              27% of potential cost
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="usage">
        <TabsList>
          <TabsTrigger value="usage">Usage Trends</TabsTrigger>
          <TabsTrigger value="providers">Provider Breakdown</TabsTrigger>
          <TabsTrigger value="costs">Cost Analysis</TabsTrigger>
        </TabsList>
        <TabsContent value="usage" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Usage Over Time</CardTitle>
              <CardDescription>
                Request and token usage over the past 30 days
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[300px]">
              <UsageChart />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="providers" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Provider Usage</CardTitle>
              <CardDescription>
                Breakdown of usage by LLM provider
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ProviderUsageTable />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="costs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Cost Breakdown</CardTitle>
              <CardDescription>
                Analysis of costs by provider and model
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CostBreakdown />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
