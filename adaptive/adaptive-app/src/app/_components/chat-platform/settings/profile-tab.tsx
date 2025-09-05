"use client";

import type { UserResource } from "@clerk/types";
import { User } from "lucide-react";
import type React from "react";
import { useId } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/trpc/react";
import type { UserMetadata, UserPreferences } from "@/types/settings";

interface ProfileTabProps {
	user: UserResource | null | undefined;
}

export const ProfileTab: React.FC<ProfileTabProps> = ({ user }) => {
	const { data: preferences } = api.user.getPreferences.useQuery();

	const fullName = preferences?.fullName || user?.fullName || "";
	const preferredName = preferences?.preferredName || user?.firstName || "";
	const jobRole = preferences?.jobRole || "";
	const personalPreferences = preferences?.personalPreferences || "";

	const fullNameId = useId();
	const preferredNameId = useId();
	const jobRoleId = useId();
	const emailId = useId();
	const personalPreferencesId = useId();

	const utils = api.useUtils();
	const updateMetadataMutation = api.user.updateMetadata.useMutation({
		onMutate: async (newMetadata: UserMetadata) => {
			await utils.user.getPreferences.cancel();

			const previousData = utils.user.getPreferences.getData();

			if (previousData) {
				utils.user.getPreferences.setData(
					undefined,
					(old: UserPreferences | undefined) => {
						if (!old) return old;
						return {
							...old,
							...newMetadata,
						};
					},
				);
			}

			return { previousData };
		},
		onError: (_err, _newMetadata, context) => {
			if (context?.previousData) {
				utils.user.getPreferences.setData(undefined, context.previousData);
			}
		},
		onSettled: () => {
			utils.user.getPreferences.invalidate();
		},
	});

	const updateMetadata = async (metadata: UserMetadata) => {
		try {
			await updateMetadataMutation.mutateAsync(metadata);
		} catch (error) {
			console.error("Error updating metadata:", error);
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
							<Label htmlFor={fullNameId}>Full Name</Label>
							<Input
								id={fullNameId}
								value={fullName as string}
								onChange={(e) => updateMetadata({ fullName: e.target.value })}
								placeholder="Enter your full name"
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor={preferredNameId}>What should we call you?</Label>
							<Input
								id={preferredNameId}
								value={preferredName as string}
								onChange={(e) =>
									updateMetadata({ preferredName: e.target.value })
								}
								placeholder="Enter your preferred name"
							/>
						</div>
					</div>

					<div className="grid grid-cols-2 gap-4">
						<div className="space-y-2">
							<Label htmlFor={jobRoleId}>Job Role</Label>
							<Input
								id={jobRoleId}
								value={jobRole as string}
								onChange={(e) => updateMetadata({ jobRole: e.target.value })}
								placeholder="e.g., Software Engineer, Designer, Manager"
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor={emailId}>Email (Read-only)</Label>
							<Input
								id={emailId}
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

					<div className="space-y-2">
						<Label htmlFor={personalPreferencesId}>
							Personal Preferences for Adaptive
						</Label>
						<Textarea
							id={personalPreferencesId}
							value={personalPreferences as string}
							onChange={(e) =>
								updateMetadata({ personalPreferences: e.target.value })
							}
							placeholder="Tell Adaptive about your work style, communication preferences, or anything else that would help personalize your experience..."
							rows={4}
						/>
						<p className="text-muted-foreground text-xs">
							This helps Adaptive understand how to best assist you
						</p>
					</div>

					<Separator />
				</CardContent>
			</Card>
		</div>
	);
};
