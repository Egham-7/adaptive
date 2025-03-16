"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertCircle, Bell } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function UsageLimits() {
  const [dailyLimit, setDailyLimit] = useState(10000);
  const [monthlyLimit, setMonthlyLimit] = useState(300000);
  const [costLimit, setCostLimit] = useState(500);
  const [alertThreshold, setAlertThreshold] = useState(80);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Usage Limits & Alerts</h2>
      </div>

      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Usage Alert</AlertTitle>
        <AlertDescription>
          You have reached 85% of your monthly token usage limit. Consider
          adjusting your limits or optimizing your usage.
        </AlertDescription>
      </Alert>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Token Usage Limits</CardTitle>
            <CardDescription>
              Set limits on token usage to control costs
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="daily-limit">Daily Token Limit</Label>
                <span className="text-sm font-medium">
                  {dailyLimit.toLocaleString()} tokens
                </span>
              </div>
              <Slider
                id="daily-limit"
                min={1000}
                max={100000}
                step={1000}
                defaultValue={[dailyLimit]}
                onValueChange={(value) => setDailyLimit(value[0])}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>1K</span>
                <span>100K</span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="monthly-limit">Monthly Token Limit</Label>
                <span className="text-sm font-medium">
                  {monthlyLimit.toLocaleString()} tokens
                </span>
              </div>
              <Slider
                id="monthly-limit"
                min={10000}
                max={1000000}
                step={10000}
                defaultValue={[monthlyLimit]}
                onValueChange={(value) => setMonthlyLimit(value[0])}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>10K</span>
                <span>1M</span>
              </div>
            </div>

            <div className="flex items-center space-x-2 pt-2">
              <Switch id="hard-limit" />
              <Label htmlFor="hard-limit">
                Enforce hard limits (block requests when limit is reached)
              </Label>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cost Limits</CardTitle>
            <CardDescription>Set maximum spending limits</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="cost-limit">Monthly Cost Limit</Label>
                <span className="text-sm font-medium">${costLimit}</span>
              </div>
              <Slider
                id="cost-limit"
                min={10}
                max={1000}
                step={10}
                defaultValue={[costLimit]}
                onValueChange={(value) => setCostLimit(value[0])}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>$10</span>
                <span>$1,000</span>
              </div>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="billing-cycle">Billing Cycle</Label>
              <Select defaultValue="calendar">
                <SelectTrigger id="billing-cycle">
                  <SelectValue placeholder="Select billing cycle" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="calendar">Calendar Month</SelectItem>
                  <SelectItem value="rolling">Rolling 30 Days</SelectItem>
                  <SelectItem value="custom">Custom Period</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center space-x-2 pt-2">
              <Switch id="auto-pause" />
              <Label htmlFor="auto-pause">
                Automatically pause API access when cost limit is reached
              </Label>
            </div>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Alert Settings</CardTitle>
            <CardDescription>
              Configure notifications for usage and cost thresholds
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="alert-threshold">Alert Threshold</Label>
                    <span className="text-sm font-medium">
                      {alertThreshold}% of limit
                    </span>
                  </div>
                  <Slider
                    id="alert-threshold"
                    min={50}
                    max={95}
                    step={5}
                    defaultValue={[alertThreshold]}
                    onValueChange={(value) => setAlertThreshold(value[0])}
                  />
                  <p className="text-xs text-muted-foreground">
                    You will receive alerts when usage reaches this percentage
                    of your set limits
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Alert Types</Label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Switch id="email-alerts" defaultChecked />
                      <Label htmlFor="email-alerts">Email Alerts</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch id="dashboard-alerts" defaultChecked />
                      <Label htmlFor="dashboard-alerts">Dashboard Alerts</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch id="webhook-alerts" />
                      <Label htmlFor="webhook-alerts">
                        Webhook Notifications
                      </Label>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="grid gap-2">
                  <Label htmlFor="alert-email">Alert Email</Label>
                  <Input
                    id="alert-email"
                    placeholder="Enter email address"
                    defaultValue="admin@example.com"
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="webhook-url">Webhook URL (Optional)</Label>
                  <Input
                    id="webhook-url"
                    placeholder="https://your-service.com/webhook"
                  />
                </div>

                <div className="flex items-center space-x-2 pt-2">
                  <Switch id="daily-summary" defaultChecked />
                  <Label htmlFor="daily-summary">
                    Send daily usage summary
                  </Label>
                </div>

                <Button className="mt-4 w-full">
                  <Bell className="mr-2 h-4 w-4" />
                  Test Notifications
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
