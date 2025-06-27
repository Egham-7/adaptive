"use client";

import { type UserResource } from "@clerk/types";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { User, Save } from "lucide-react";
import type React from "react";
import { api } from "@/trpc/react";

interface ProfileTabProps {
  user: UserResource | null | undefined;
}

export const ProfileTab: React.FC<ProfileTabProps> = ({ user }) => {
  const displayName = user?.publicMetadata?.displayName || user?.fullName || "";
  const responseStyle = user?.publicMetadata?.responseStyle || "balanced";
  const language = user?.publicMetadata?.language || "english";

  const updateMetadataMutation = api.user.updateMetadata.useMutation();

  const updateMetadata = async (metadata: Record<string, any>) => {
    try {
      await updateMetadataMutation.mutateAsync(metadata);
    } catch (error) {
      console.error('Error updating metadata:', error);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5" />
            User Profile
          </CardTitle>
          <CardDescription>Personalize your AI chat experience</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center gap-4">
            <Avatar className="h-20 w-20">
              <AvatarImage src={user?.imageUrl} />
              <AvatarFallback className="text-lg">
                {user?.firstName?.[0]}
                {user?.lastName?.[0]}
              </AvatarFallback>
            </Avatar>
            <div className="space-y-2">
              <div className="space-y-1">
                <h3 className="font-semibold text-xl">
                  {user?.fullName || "User"}
                </h3>
                <p className="text-muted-foreground">
                  {user?.primaryEmailAddress?.emailAddress}
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="displayName">Display Name</Label>
              <Input
                id="displayName"
                value={displayName as string}
                readOnly
                className="bg-muted"
                placeholder={user?.fullName || "Enter your display name"}
              />
              <p className="text-muted-foreground text-xs">
                Manage display name through server actions
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email (Read-only)</Label>
              <Input
                id="email"
                type="email"
                value={user?.primaryEmailAddress?.emailAddress || ""}
                disabled
                className="bg-muted"
              />
              <p className="text-muted-foreground text-xs">
                Manage email in Account settings
              </p>
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <h3 className="font-medium text-lg">AI Preferences</h3>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Response Style</Label>
                <Select value={responseStyle as string} disabled>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="concise">Concise</SelectItem>
                    <SelectItem value="balanced">Balanced</SelectItem>
                    <SelectItem value="detailed">Detailed</SelectItem>
                    <SelectItem value="creative">Creative</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Language</Label>
                <Select
                  value={(language as string) || "english"}
                  onValueChange={(value) => updateMetadata({ language: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="english">English</SelectItem>
                    <SelectItem value="spanish">Spanish</SelectItem>
                    <SelectItem value="french">French</SelectItem>
                    <SelectItem value="german">German</SelectItem>
                    <SelectItem value="chinese">Chinese</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <Button
              onClick={() => updateMetadata({})}
              className="flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              Save Preferences
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
