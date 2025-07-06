"use client";

import { useUser } from "@clerk/nextjs";
import { Brain, CreditCard, Settings, Target, User } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { PlansTab } from "@/app/_components/chat-platform/settings/plans-tab";
import { PreferencesTab } from "@/app/_components/chat-platform/settings/preferences-tab";
import { ProfileTab } from "@/app/_components/chat-platform/settings/profile-tab";
import { ProvidersTab } from "@/app/_components/chat-platform/settings/providers-tab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/trpc/react";
import type { Provider, UserPreferences } from "@/types/settings";

const SettingsPage: React.FC = () => {
	const { user, isLoaded } = useUser();
	const searchParams = useSearchParams();
	const activeTab = searchParams.get("tab") || "providers"; // Default to "providers"

	const { data: preferences, isLoading } = api.user.getPreferences.useQuery(
		undefined,
		{ enabled: isLoaded && !!user },
	);
	const utils = api.useUtils();
	const updatePreferencesMutation = api.user.updatePreferences.useMutation({
		onMutate: async (newPreferences: { providers: Provider[] }) => {
			await utils.user.getPreferences.cancel();

			const previousData = utils.user.getPreferences.getData();

			if (previousData) {
				utils.user.getPreferences.setData(
					undefined,
					(old: UserPreferences | undefined) => {
						if (!old) return old;
						return {
							...old,
							...newPreferences,
						};
					},
				);
			}

			return { previousData };
		},
		onError: (_err, _newPreferences, context) => {
			if (context?.previousData) {
				utils.user.getPreferences.setData(undefined, context.previousData);
			}
		},
		onSettled: () => {
			utils.user.getPreferences.invalidate();
		},
	});

	const updateProvider = (id: string, updates: Partial<Provider>) => {
		if (!preferences) return;

		const newProviders = preferences.providers.map((p: Provider) =>
			p.id === id ? { ...p, ...updates } : p,
		);

		updatePreferencesMutation.mutate({ providers: newProviders });
	};

	// Don't render anything if not loaded yet - loading.tsx will handle this
	if (!isLoaded || isLoading) {
		return null;
	}

	return (
		<div className="min-h-screen bg-background p-6">
			<div className="mx-auto max-w-6xl space-y-6">
				<div className="mb-8 flex items-center gap-3">
					<Settings className="h-8 w-8 text-primary" />
					<h1 className="font-bold text-3xl text-foreground">Settings</h1>
				</div>

				<Tabs defaultValue={activeTab} className="space-y-6">
					<TabsList className="flex w-full flex-wrap gap-2">
						<TabsTrigger
							value="providers"
							className="flex items-center gap-2 truncate"
						>
							<Brain className="h-4 w-4 shrink-0" />
							Providers
						</TabsTrigger>

						<TabsTrigger
							value="profile"
							className="flex items-center gap-2 truncate"
						>
							<User className="h-4 w-4 shrink-0" />
							Profile
						</TabsTrigger>

						<TabsTrigger
							value="preferences"
							className="flex items-center gap-2 truncate"
						>
							<Target className="h-4 w-4 shrink-0" />
							Preferences
						</TabsTrigger>

						<TabsTrigger
							value="plans"
							className="flex items-center gap-2 truncate"
						>
							<CreditCard className="h-4 w-4 shrink-0" />
							Plans
						</TabsTrigger>
					</TabsList>

					<TabsContent value="providers">
						<ProvidersTab
							providers={preferences?.providers || []}
							updateProvider={updateProvider}
						/>
					</TabsContent>

					<TabsContent value="profile">
						<ProfileTab user={user} />
					</TabsContent>

					<TabsContent value="preferences">
						<PreferencesTab />
					</TabsContent>

					<TabsContent value="plans">
						<PlansTab />
					</TabsContent>
				</Tabs>
			</div>
		</div>
	);
};

export default SettingsPage;
