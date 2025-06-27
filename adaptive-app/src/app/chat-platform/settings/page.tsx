"use client";

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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
	Bell,
	Brain,
	Database,
	DollarSign,
	Save,
	Settings,
	Target,
	User,
	Zap,
} from "lucide-react";
import type React from "react";
import { useState } from "react";

interface Provider {
	id: string;
	name: string;
	enabled: boolean;
	priority: number;
	costPerToken: number;
	qualityScore: number;
}

interface ModelConfig {
	id: string;
	name: string;
	provider: string;
	enabled: boolean;
	useCase: string[];
}

interface UserProfile {
	name: string;
	email: string;
	avatar: string;
	preferences: {
		responseStyle: string;
		expertise: string;
		language: string;
	};
}

interface SettingsData {
	providers: Provider[];
	models: ModelConfig[];
	userProfile: UserProfile;
	routingStrategy: string;
	autoRouting: boolean;
	notifications: {
		newMessages: boolean;
		modelSwitches: boolean;
		costAlerts: boolean;
	};
	theme: string;
	maxTokens: number;
	temperature: number;
}

const AISettingsPage: React.FC = () => {
	const [settings, setSettings] = useState<SettingsData>({
		providers: [
			{
				id: "1",
				name: "OpenAI",
				enabled: true,
				priority: 1,
				costPerToken: 0.002,
				qualityScore: 9.5,
			},
			{
				id: "2",
				name: "Anthropic",
				enabled: true,
				priority: 2,
				costPerToken: 0.003,
				qualityScore: 9.2,
			},
			{
				id: "3",
				name: "Gemini",
				enabled: false,
				priority: 3,
				costPerToken: 0.001,
				qualityScore: 8.8,
			},
			{
				id: "4",
				name: "Groq",
				enabled: true,
				priority: 4,
				costPerToken: 0.0025,
				qualityScore: 8.7,
			},
			{
				id: "5",
				name: "Deepseek",
				enabled: true,
				priority: 5,
				costPerToken: 0.0018,
				qualityScore: 8.3,
			},
			{
				id: "6",
				name: "HuggingFace",
				enabled: true,
				priority: 6,
				costPerToken: 0.0022,
				qualityScore: 8.9,
			},
		],
		models: [
			{
				id: "1",
				name: "GPT-4",
				provider: "OpenAI",
				enabled: true,
				useCase: ["complex reasoning", "creative writing"],
			},
			{
				id: "2",
				name: "GPT-3.5 Turbo",
				provider: "OpenAI",
				enabled: true,
				useCase: ["general chat", "quick responses"],
			},
			{
				id: "3",
				name: "Claude 3",
				provider: "Anthropic",
				enabled: true,
				useCase: ["analysis", "research"],
			},
			{
				id: "4",
				name: "Gemini Pro",
				provider: "Gemini",
				enabled: false,
				useCase: ["multimodal", "coding"],
			},
		],
		userProfile: {
			name: "John Doe",
			email: "john.doe@example.com",
			avatar: "",
			preferences: {
				responseStyle: "balanced",
				expertise: "intermediate",
				language: "english",
			},
		},
		routingStrategy: "balanced",
		autoRouting: true,
		notifications: {
			newMessages: true,
			modelSwitches: false,
			costAlerts: true,
		},
		theme: "system",
		maxTokens: 2048,
		temperature: 0.7,
	});

	const updateProvider = (id: string, updates: Partial<Provider>) => {
		setSettings((prev) => ({
			...prev,
			providers: prev.providers.map((p) =>
				p.id === id ? { ...p, ...updates } : p,
			),
		}));
	};

	const updateModel = (id: string, updates: Partial<ModelConfig>) => {
		setSettings((prev) => ({
			...prev,
			models: prev.models.map((m) => (m.id === id ? { ...m, ...updates } : m)),
		}));
	};

	const updateUserProfile = (updates: Partial<UserProfile>) => {
		setSettings((prev) => ({
			...prev,
			userProfile: { ...prev.userProfile, ...updates },
		}));
	};

	const updateNotifications = (
		key: keyof typeof settings.notifications,
		value: boolean,
	) => {
		setSettings((prev) => ({
			...prev,
			notifications: { ...prev.notifications, [key]: value },
		}));
	};

	return (
		<div className="min-h-screen bg-background p-6">
			<div className="mx-auto max-w-6xl space-y-6">
				<div className="mb-8 flex items-center gap-3">
					<Settings className="h-8 w-8 text-primary" />
					<h1 className="font-bold text-3xl text-foreground">
						AI Chat Settings
					</h1>
				</div>

				<Tabs defaultValue="routing" className="space-y-6">
					<TabsList className="grid w-full grid-cols-5">
						<TabsTrigger value="routing" className="flex items-center gap-2">
							<Brain className="h-4 w-4" />
							Routing & Providers
						</TabsTrigger>
						<TabsTrigger value="profile" className="flex items-center gap-2">
							<User className="h-4 w-4" />
							Profile
						</TabsTrigger>
						<TabsTrigger
							value="preferences"
							className="flex items-center gap-2"
						>
							<Target className="h-4 w-4" />
							Preferences
						</TabsTrigger>
						<TabsTrigger value="advanced" className="flex items-center gap-2">
							<Settings className="h-4 w-4" />
							Advanced
						</TabsTrigger>
						<TabsTrigger
							value="notifications"
							className="flex items-center gap-2"
						>
							<Bell className="h-4 w-4" />
							Notifications
						</TabsTrigger>
					</TabsList>

					{/* Routing & Providers */}
					<TabsContent value="routing" className="space-y-6">
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<Brain className="h-5 w-5" />
									Intelligent Model Routing
								</CardTitle>
								<CardDescription>
									Configure how the AI automatically selects models and which
									providers to route against
								</CardDescription>
							</CardHeader>
							<CardContent className="space-y-6">
								<div className="flex items-center justify-between">
									<div>
										<Label className="font-medium text-base">
											Auto Routing
										</Label>
										<p className="text-muted-foreground text-sm">
											Automatically select the best model for each query
										</p>
									</div>
									<Switch
										checked={settings.autoRouting}
										onCheckedChange={(checked) =>
											setSettings((prev) => ({
												...prev,
												autoRouting: checked,
											}))
										}
									/>
								</div>

								<Separator />

								<div className="space-y-4">
									<Label className="font-medium text-base">
										Routing Strategy
									</Label>
									<RadioGroup
										value={settings.routingStrategy}
										onValueChange={(value) =>
											setSettings((prev) => ({
												...prev,
												routingStrategy: value,
											}))
										}
										className="grid grid-cols-3 gap-4"
									>
										<div className="flex items-center space-x-2 rounded-lg border p-4">
											<RadioGroupItem value="cost" id="cost" />
											<div className="flex-1">
												<Label
													htmlFor="cost"
													className="flex items-center gap-2 font-medium"
												>
													<DollarSign className="h-4 w-4 text-green-500" />
													Cost Optimized
												</Label>
												<p className="text-muted-foreground text-xs">
													Prioritize cheaper models
												</p>
											</div>
										</div>
										<div className="flex items-center space-x-2 rounded-lg border p-4">
											<RadioGroupItem value="balanced" id="balanced" />
											<div className="flex-1">
												<Label
													htmlFor="balanced"
													className="flex items-center gap-2 font-medium"
												>
													<Target className="h-4 w-4 text-blue-500" />
													Balanced
												</Label>
												<p className="text-muted-foreground text-xs">
													Balance cost and quality
												</p>
											</div>
										</div>
										<div className="flex items-center space-x-2 rounded-lg border p-4">
											<RadioGroupItem value="quality" id="quality" />
											<div className="flex-1">
												<Label
													htmlFor="quality"
													className="flex items-center gap-2 font-medium"
												>
													<Zap className="h-4 w-4 text-purple-500" />
													Quality First
												</Label>
												<p className="text-muted-foreground text-xs">
													Prioritize best models
												</p>
											</div>
										</div>
									</RadioGroup>
								</div>
							</CardContent>
						</Card>

						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<Database className="h-5 w-5" />
									AI Providers
								</CardTitle>
								<CardDescription>
									Manage your AI provider connections and routing preferences
								</CardDescription>
							</CardHeader>
							<CardContent>
								<div className="grid grid-cols-2 gap-4 md:grid-cols-4">
									{settings.providers.map((provider) => (
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
														<img
															src={`/logos/${provider.name
																.toLowerCase()
																.replace(" ", "")}.svg`}
															alt={provider.name}
															className="h-10 w-10 object-contain"
														/>
													</div>
													<div className="text-center">
														<div className="font-medium text-sm">
															{provider.name}
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
					</TabsContent>

					{/* Profile */}
					<TabsContent value="profile" className="space-y-6">
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<User className="h-5 w-5" />
									User Profile
								</CardTitle>
								<CardDescription>
									Personalize your AI chat experience
								</CardDescription>
							</CardHeader>
							<CardContent className="space-y-6">
								<div className="flex items-center gap-4">
									<Avatar className="h-20 w-20">
										<AvatarImage src={settings.userProfile.avatar} />
										<AvatarFallback className="text-lg">
											{settings.userProfile.name
												.split(" ")
												.map((n) => n[0])
												.join("")}
										</AvatarFallback>
									</Avatar>
									<div className="space-y-2">
										<Button variant="outline" size="sm">
											Change Avatar
										</Button>
										<p className="text-muted-foreground text-sm">
											JPG, PNG or GIF. Max 2MB.
										</p>
									</div>
								</div>

								<div className="grid grid-cols-2 gap-4">
									<div className="space-y-2">
										<Label htmlFor="name">Full Name</Label>
										<Input
											id="name"
											value={settings.userProfile.name}
											onChange={(e) =>
												updateUserProfile({ name: e.target.value })
											}
										/>
									</div>
									<div className="space-y-2">
										<Label htmlFor="email">Email</Label>
										<Input
											id="email"
											type="email"
											value={settings.userProfile.email}
											onChange={(e) =>
												updateUserProfile({ email: e.target.value })
											}
										/>
									</div>
								</div>

								<Separator />

								<div className="space-y-4">
									<h3 className="font-medium text-lg">AI Preferences</h3>

									<div className="grid grid-cols-3 gap-4">
										<div className="space-y-2">
											<Label>Response Style</Label>
											<Select
												value={settings.userProfile.preferences.responseStyle}
												onValueChange={(value) =>
													updateUserProfile({
														preferences: {
															...settings.userProfile.preferences,
															responseStyle: value,
														},
													})
												}
											>
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
											<Label>Expertise Level</Label>
											<Select
												value={settings.userProfile.preferences.expertise}
												onValueChange={(value) =>
													updateUserProfile({
														preferences: {
															...settings.userProfile.preferences,
															expertise: value,
														},
													})
												}
											>
												<SelectTrigger>
													<SelectValue />
												</SelectTrigger>
												<SelectContent>
													<SelectItem value="beginner">Beginner</SelectItem>
													<SelectItem value="intermediate">
														Intermediate
													</SelectItem>
													<SelectItem value="advanced">Advanced</SelectItem>
													<SelectItem value="expert">Expert</SelectItem>
												</SelectContent>
											</Select>
										</div>

										<div className="space-y-2">
											<Label>Language</Label>
											<Select
												value={settings.userProfile.preferences.language}
												onValueChange={(value) =>
													updateUserProfile({
														preferences: {
															...settings.userProfile.preferences,
															language: value,
														},
													})
												}
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
							</CardContent>
						</Card>
					</TabsContent>

					{/* Preferences */}
					<TabsContent value="preferences" className="space-y-6">
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<Target className="h-5 w-5" />
									Chat Preferences
								</CardTitle>
								<CardDescription>
									Customize your chat experience
								</CardDescription>
							</CardHeader>
							<CardContent className="space-y-6">
								<div className="space-y-4">
									<h3 className="font-medium text-lg">Response Style</h3>

									<div className="space-y-2">
										<Label>How creative should responses be?</Label>
										<RadioGroup
											value={
												settings.temperature > 0.8
													? "creative"
													: settings.temperature > 0.4
														? "balanced"
														: "focused"
											}
											onValueChange={(value) => {
												const temp =
													value === "creative"
														? 0.9
														: value === "balanced"
															? 0.7
															: 0.3;
												setSettings((prev) => ({
													...prev,
													temperature: temp,
												}));
											}}
											className="grid grid-cols-3 gap-4"
										>
											<div className="flex items-center space-x-2 rounded-lg border p-4">
												<RadioGroupItem value="focused" id="focused" />
												<div className="flex-1">
													<Label htmlFor="focused" className="font-medium">
														Focused
													</Label>
													<p className="text-muted-foreground text-xs">
														Precise and factual responses
													</p>
												</div>
											</div>
											<div className="flex items-center space-x-2 rounded-lg border p-4">
												<RadioGroupItem value="balanced" id="balanced-style" />
												<div className="flex-1">
													<Label
														htmlFor="balanced-style"
														className="font-medium"
													>
														Balanced
													</Label>
													<p className="text-muted-foreground text-xs">
														Mix of creativity and accuracy
													</p>
												</div>
											</div>
											<div className="flex items-center space-x-2 rounded-lg border p-4">
												<RadioGroupItem value="creative" id="creative" />
												<div className="flex-1">
													<Label htmlFor="creative" className="font-medium">
														Creative
													</Label>
													<p className="text-muted-foreground text-xs">
														More imaginative responses
													</p>
												</div>
											</div>
										</RadioGroup>
									</div>
								</div>

								<Separator />

								<div className="space-y-4">
									<h3 className="font-medium text-lg">Response Length</h3>

									<RadioGroup
										value={
											settings.maxTokens > 2048
												? "long"
												: settings.maxTokens > 1024
													? "medium"
													: "short"
										}
										onValueChange={(value) => {
											const tokens =
												value === "long"
													? 4096
													: value === "medium"
														? 2048
														: 512;
											setSettings((prev) => ({ ...prev, maxTokens: tokens }));
										}}
										className="grid grid-cols-3 gap-4"
									>
										<div className="flex items-center space-x-2 rounded-lg border p-4">
											<RadioGroupItem value="short" id="short" />
											<div className="flex-1">
												<Label htmlFor="short" className="font-medium">
													Short
												</Label>
												<p className="text-muted-foreground text-xs">
													Quick, concise answers
												</p>
											</div>
										</div>
										<div className="flex items-center space-x-2 rounded-lg border p-4">
											<RadioGroupItem value="medium" id="medium" />
											<div className="flex-1">
												<Label htmlFor="medium" className="font-medium">
													Medium
												</Label>
												<p className="text-muted-foreground text-xs">
													Detailed explanations
												</p>
											</div>
										</div>
										<div className="flex items-center space-x-2 rounded-lg border p-4">
											<RadioGroupItem value="long" id="long" />
											<div className="flex-1">
												<Label htmlFor="long" className="font-medium">
													Long
												</Label>
												<p className="text-muted-foreground text-xs">
													Comprehensive responses
												</p>
											</div>
										</div>
									</RadioGroup>
								</div>

								<Separator />

								<div className="space-y-4">
									<h3 className="font-medium text-lg">Theme & Appearance</h3>

									<div className="space-y-2">
										<Label>Theme</Label>
										<Select
											value={settings.theme}
											onValueChange={(value) =>
												setSettings((prev) => ({ ...prev, theme: value }))
											}
										>
											<SelectTrigger className="w-48">
												<SelectValue />
											</SelectTrigger>
											<SelectContent>
												<SelectItem value="light">Light</SelectItem>
												<SelectItem value="dark">Dark</SelectItem>
												<SelectItem value="system">System</SelectItem>
											</SelectContent>
										</Select>
									</div>
								</div>
							</CardContent>
						</Card>
					</TabsContent>

					{/* Advanced */}
					<TabsContent value="advanced" className="space-y-6">
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<Settings className="h-5 w-5" />
									Advanced Settings
								</CardTitle>
								<CardDescription>
									Technical configuration for power users
								</CardDescription>
							</CardHeader>
							<CardContent className="space-y-6">
								<div className="grid grid-cols-2 gap-6">
									<div className="space-y-2">
										<Label>Max Tokens per Response</Label>
										<div className="px-3">
											<Slider
												value={[settings.maxTokens]}
												onValueChange={([value]) =>
													setSettings((prev) => ({
														...prev,
														maxTokens: value ?? 4096,
													}))
												}
												max={4096}
												min={256}
												step={256}
												className="w-full"
											/>
											<div className="mt-1 flex justify-between text-muted-foreground text-xs">
												<span>256</span>
												<span>{settings.maxTokens}</span>
												<span>4096</span>
											</div>
										</div>
									</div>

									<div className="space-y-2">
										<Label>Temperature (Creativity)</Label>
										<div className="px-3">
											<Slider
												value={[settings.temperature * 100]}
												onValueChange={([value = 50]) =>
													setSettings((prev) => ({
														...prev,
														temperature: value / 100,
													}))
												}
												max={100}
												min={0}
												step={5}
												className="w-full"
											/>
											<div className="mt-1 flex justify-between text-muted-foreground text-xs">
												<span>0</span>
												<span>{Math.round(settings.temperature * 100)}%</span>
												<span>100</span>
											</div>
										</div>
									</div>
								</div>
							</CardContent>
						</Card>
					</TabsContent>

					{/* Notifications */}
					<TabsContent value="notifications" className="space-y-6">
						<Card>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<Bell className="h-5 w-5" />
									Notifications
								</CardTitle>
								<CardDescription>
									Configure when and how you want to be notified
								</CardDescription>
							</CardHeader>
							<CardContent className="space-y-6">
								<div className="space-y-4">
									<div className="flex items-center justify-between">
										<div>
											<Label className="font-medium text-base">
												New Messages
											</Label>
											<p className="text-muted-foreground text-sm">
												Get notified when you receive new chat messages
											</p>
										</div>
										<Switch
											checked={settings.notifications.newMessages}
											onCheckedChange={(checked) =>
												updateNotifications("newMessages", checked)
											}
										/>
									</div>

									<div className="flex items-center justify-between">
										<div>
											<Label className="font-medium text-base">
												Model Switches
											</Label>
											<p className="text-muted-foreground text-sm">
												Notify when the AI switches to a different model
											</p>
										</div>
										<Switch
											checked={settings.notifications.modelSwitches}
											onCheckedChange={(checked) =>
												updateNotifications("modelSwitches", checked)
											}
										/>
									</div>

									<div className="flex items-center justify-between">
										<div>
											<Label className="font-medium text-base">
												Cost Alerts
											</Label>
											<p className="text-muted-foreground text-sm">
												Alert when approaching your cost threshold
											</p>
										</div>
										<Switch
											checked={settings.notifications.costAlerts}
											onCheckedChange={(checked) =>
												updateNotifications("costAlerts", checked)
											}
										/>
									</div>
								</div>
							</CardContent>
						</Card>
					</TabsContent>
				</Tabs>

				<div className="flex justify-end gap-3 border-t pt-6">
					<Button variant="outline">Reset to Defaults</Button>
					<Button className="flex items-center gap-2">
						<Save className="h-4 w-4" />
						Save Settings
					</Button>
				</div>
			</div>
		</div>
	);
};

export default AISettingsPage;
