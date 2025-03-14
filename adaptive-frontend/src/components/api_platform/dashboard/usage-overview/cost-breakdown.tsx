import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Card, CardContent } from "@/components/ui/card";

const data = [
  { name: "OpenAI", value: 240.5, color: "#0ea5e9" },
  { name: "Anthropic", value: 150.75, color: "#8b5cf6" },
  { name: "Mistral AI", value: 90.25, color: "#22c55e" },
  { name: "Cohere", value: 50.4, color: "#f59e0b" },
];

export function CostBreakdown() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <Card>
        <CardContent className="p-6">
          <h3 className="mb-4 text-lg font-medium">Cost by Provider</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                  labelLine={false}
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`$${value}`, "Cost"]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <Card>
          <CardContent className="p-6">
            <h3 className="mb-2 text-lg font-medium">Cost Optimization</h3>
            <div className="space-y-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Actual Cost</span>
                  <span className="font-medium">$531.90</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">
                    Potential Cost (without Adaptive)
                  </span>
                  <span className="font-medium">$660.30</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Total Savings</span>
                  <span className="font-medium text-green-600">
                    $128.40 (19.4%)
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <h3 className="mb-2 text-lg font-medium">
              Cost Efficiency by Model Type
            </h3>
            <div className="space-y-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Completion Requests</span>
                  <span className="font-medium">$0.0023 / 1K tokens</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Chat Completions</span>
                  <span className="font-medium">$0.0027 / 1K tokens</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Embeddings</span>
                  <span className="font-medium">$0.0001 / 1K tokens</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
