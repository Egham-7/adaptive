"use client";

import { Edit, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface ApiKey {
	id: string;
	name: string;
	key: string;
	created: string;
	lastUsed: string;
	createdBy: string;
	permissions: string;
}

const initialApiKeys: ApiKey[] = [
	{
		id: "1",
		name: "production-api",
		key: "sk-...wzKA",
		created: "Jun 22, 2025",
		lastUsed: "Jun 23, 2025",
		createdBy: "Sarah Chen",
		permissions: "All",
	},
	{
		id: "2",
		name: "development-env",
		key: "sk-...634A",
		created: "Jun 20, 2025",
		lastUsed: "Jun 20, 2025",
		createdBy: "Sarah Chen",
		permissions: "All",
	},
	{
		id: "3",
		name: "analytics-service",
		key: "sk-...Rr8A",
		created: "Jun 15, 2025",
		lastUsed: "Jun 22, 2025",
		createdBy: "Sarah Chen",
		permissions: "All",
	},
	{
		id: "4",
		name: "mobile-app",
		key: "sk-...WIA",
		created: "Jun 4, 2025",
		lastUsed: "Jun 24, 2025",
		createdBy: "Sarah Chen",
		permissions: "All",
	},
	{
		id: "5",
		name: "data-pipeline",
		key: "sk-...5k4A",
		created: "May 20, 2025",
		lastUsed: "Jun 3, 2025",
		createdBy: "Sarah Chen",
		permissions: "All",
	},
];

export default function ApiKeysPage() {
	const [apiKeys, setApiKeys] = useState<ApiKey[]>(initialApiKeys);
	const [showCreateDialog, setShowCreateDialog] = useState(false);
	const [formData, setFormData] = useState({
		name: "",
		description: "",
	});

	const resetForm = () => {
		setFormData({
			name: "",
			description: "",
		});
	};

	const generateApiKey = () => {
		const randomKey = Array.from(
			{ length: 4 },
			() =>
				String.fromCharCode(65 + Math.floor(Math.random() * 26)) +
				String.fromCharCode(97 + Math.floor(Math.random() * 26)) +
				Math.floor(Math.random() * 10),
		).join("");
		return `sk-...${randomKey.slice(0, 4)}`;
	};

	const handleCreateApiKey = () => {
		const newKey: ApiKey = {
			id: Date.now().toString(),
			name: formData.name,
			key: generateApiKey(),
			created: new Date().toLocaleDateString("en-US", {
				year: "numeric",
				month: "short",
				day: "numeric",
			}),
			lastUsed: "Never",
			createdBy: "Sarah Chen",
			permissions: "All",
		};

		setApiKeys([newKey, ...apiKeys]);
		setShowCreateDialog(false);
		resetForm();
	};

	const handleDeleteApiKey = (id: string) => {
		setApiKeys(apiKeys.filter((key) => key.id !== id));
	};

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<h1 className="font-bold text-2xl text-foreground">
					API keys
				</h1>
				<Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
					<DialogTrigger asChild>
						<Button className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
							<Plus className="h-4 w-4" />
							Create new secret key
						</Button>
					</DialogTrigger>
					<DialogContent className="max-w-md">
						<DialogHeader>
							<DialogTitle>Create new secret key</DialogTitle>
						</DialogHeader>
						<div className="space-y-4">
							<div>
								<Label htmlFor="name">Name</Label>
								<Input
									id="name"
									value={formData.name}
									onChange={(e) =>
										setFormData({ ...formData, name: e.target.value })
									}
									placeholder="My test key"
								/>
							</div>
							<div>
								<Label htmlFor="description">Description (optional)</Label>
								<Textarea
									id="description"
									value={formData.description}
									onChange={(e) =>
										setFormData({ ...formData, description: e.target.value })
									}
									placeholder="What's this key for?"
									rows={3}
								/>
							</div>
							<div className="flex justify-end gap-2">
								<Button
									variant="outline"
									onClick={() => setShowCreateDialog(false)}
								>
									Cancel
								</Button>
								<Button
									onClick={handleCreateApiKey}
									disabled={!formData.name}
									className="bg-primary hover:bg-primary/90"
								>
									Create secret key
								</Button>
							</div>
						</div>
					</DialogContent>
				</Dialog>
			</div>

			{/* Description */}
			<div className="space-y-3 text-muted-foreground text-sm">
				<p>
					As an owner of this project, you can view and manage all API keys in
					this project.
				</p>
				<p>
					Do not share your API key with others or expose it in the browser or
					other client-side code. To protect your account's security, Adaptive
					may automatically disable any API key that has leaked publicly.
				</p>
				<p>
					View usage per API key on the{" "}
					<Button
						variant="link"
						className="h-auto p-0 text-primary hover:text-primary/80"
					>
						Usage page
					</Button>
					.
				</p>
			</div>

			{/* Table */}
			<div className="overflow-hidden rounded-lg border border-border bg-card">
				<div className="overflow-x-auto">
					<table className="w-full">
						<thead className="bg-muted/50">
							<tr>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Name
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Secret Key
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Created
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Last Used
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Created By
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Permissions
								</th>
								<th className="px-6 py-3 text-right font-medium text-gray-500 text-xs uppercase tracking-wider dark:text-gray-400">
									Actions
								</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-border">
							{apiKeys.map((apiKey) => (
								<tr
									key={apiKey.id}
									className="hover:bg-muted/50"
								>
									<td className="whitespace-nowrap px-6 py-4 font-medium text-card-foreground text-sm">
										{apiKey.name}
									</td>
									<td className="whitespace-nowrap px-6 py-4 font-mono text-muted-foreground text-sm">
										{apiKey.key}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{apiKey.created}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{apiKey.lastUsed}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{apiKey.createdBy}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{apiKey.permissions}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-right font-medium text-sm">
										<div className="flex items-center justify-end gap-2">
											<Button
												variant="ghost"
												size="sm"
												className="h-auto p-1 text-muted-foreground hover:text-foreground"
											>
												<Edit className="h-4 w-4" />
											</Button>
											<Button
												variant="ghost"
												size="sm"
												onClick={() => handleDeleteApiKey(apiKey.id)}
												className="h-auto p-1 text-muted-foreground hover:text-destructive"
											>
												<Trash2 className="h-4 w-4" />
											</Button>
										</div>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</div>

			{apiKeys.length === 0 && (
				<div className="py-12 text-center">
					<h3 className="mb-2 font-medium text-foreground text-lg">
						No API keys
					</h3>
					<p className="mb-4 text-muted-foreground">
						Create your first API key to get started with the Adaptive API.
					</p>
					<Button
						onClick={() => setShowCreateDialog(true)}
						className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
					>
						<Plus className="h-4 w-4" />
						Create new secret key
					</Button>
				</div>
			)}
		</div>
	);
}
