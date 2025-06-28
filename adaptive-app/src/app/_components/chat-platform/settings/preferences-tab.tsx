"use client";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Target } from "lucide-react";
import { useTheme } from "next-themes";
import type React from "react";
import { useEffect, useState } from "react";

export const PreferencesTab: React.FC = () => {
	const { theme, setTheme } = useTheme();
	const [mounted, setMounted] = useState(false);

	useEffect(() => {
		setMounted(true);
	}, []);

	if (!mounted) {
		return (
			<div className="space-y-6">
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center gap-2">
							<Target className="h-5 w-5" />
							App Preferences
						</CardTitle>
						<CardDescription>
							Customize your application experience
						</CardDescription>
					</CardHeader>
					<CardContent className="space-y-6">
						<div className="space-y-4">
							<h3 className="font-medium text-lg">Theme & Appearance</h3>
							<div className="space-y-2">
								<Label>Theme</Label>
								<div className="h-10 w-48 animate-pulse rounded-md bg-muted" />
							</div>
						</div>
					</CardContent>
				</Card>
			</div>
		);
	}
	return (
		<div className="space-y-6">
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2">
						<Target className="h-5 w-5" />
						App Preferences
					</CardTitle>
					<CardDescription>
						Customize your application experience
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-6">
					<div className="space-y-4">
						<h3 className="font-medium text-lg">Theme & Appearance</h3>

						<div className="space-y-2">
							<Label>Theme</Label>
							<Select value={theme} onValueChange={setTheme}>
								<SelectTrigger className="w-48">
									<SelectValue />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="light">Light</SelectItem>
									<SelectItem value="dark">Dark</SelectItem>
									<SelectItem value="system">System</SelectItem>
								</SelectContent>
							</Select>
							<p className="text-muted-foreground text-xs">
								Choose your preferred color theme
							</p>
						</div>
					</div>
				</CardContent>
			</Card>
		</div>
	);
};
