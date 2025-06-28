"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Database } from "lucide-react";
import Image from "next/image";
import { type Provider } from "@/types/settings";

interface ProvidersTabProps {
  providers: Provider[];
  updateProvider: (id: string, updates: Partial<Provider>) => void;
}

export const ProvidersTab: React.FC<ProvidersTabProps> = ({
  providers,
  updateProvider,
}) => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            AI Providers
          </CardTitle>
          <CardDescription>
            Manage your AI provider connections and preferences
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {providers.map((provider) => (
              <div key={provider.id} className="relative">
                <div
                  className={`cursor-pointer rounded-lg border-2 p-6 transition-all ${
                    provider.enabled
                      ? "border-primary bg-primary/5"
                      : "border-muted hover:border-muted-foreground/50"
                  }`}
                >
                  <div className="flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center">
                      <Image
                        src={`/logos/${provider.name
                          .toLowerCase()
                          .replace(" ", "")}.svg`}
                        alt={provider.name}
                        width={40}
                        height={40}
                        className=" object-contain"
                      />
                    </div>
                    <div className="text-center">
                      <div className="font-medium text-sm">{provider.name}</div>
                      <div className="text-xs text-muted-foreground">
                        ${provider.costPerToken}/token
                      </div>
                    </div>
                  </div>
                  <Switch
                    checked={provider.enabled}
                    onCheckedChange={(checked) =>
                      updateProvider(provider.id, {
                        enabled: checked,
                      })
                    }
                    className="absolute top-2 right-2"
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
