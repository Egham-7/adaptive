import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const providers = [
  {
    name: "OpenAI",
    requests: 23456,
    tokens: 1200000,
    cost: 240.5,
    models: ["gpt-4o", "gpt-3.5-turbo"],
    percentage: 45,
  },
  {
    name: "Anthropic",
    requests: 12345,
    tokens: 600000,
    cost: 150.75,
    models: ["claude-3-opus", "claude-3-sonnet"],
    percentage: 28,
  },
  {
    name: "Mistral AI",
    requests: 8765,
    tokens: 450000,
    cost: 90.25,
    models: ["mistral-large", "mistral-medium"],
    percentage: 17,
  },
  {
    name: "Cohere",
    requests: 5432,
    tokens: 250000,
    cost: 50.4,
    models: ["command", "command-light"],
    percentage: 10,
  },
];

export function ProviderUsageTable() {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Provider</TableHead>
          <TableHead>Requests</TableHead>
          <TableHead>Tokens</TableHead>
          <TableHead>Cost</TableHead>
          <TableHead>Models</TableHead>
          <TableHead>Distribution</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {providers.map((provider) => (
          <TableRow key={provider.name}>
            <TableCell className="font-medium">{provider.name}</TableCell>
            <TableCell>{provider.requests.toLocaleString()}</TableCell>
            <TableCell>{provider.tokens.toLocaleString()}</TableCell>
            <TableCell>${provider.cost.toFixed(2)}</TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                {provider.models.map((model) => (
                  <Badge key={model} variant="outline">
                    {model}
                  </Badge>
                ))}
              </div>
            </TableCell>
            <TableCell>
              <div className="flex items-center gap-2">
                <Progress value={provider.percentage} className="h-2 w-full" />
                <span className="w-10 text-sm">{provider.percentage}%</span>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
