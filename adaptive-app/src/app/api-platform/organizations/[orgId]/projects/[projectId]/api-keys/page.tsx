"use client";

import { Edit, Plus, Trash2 } from "lucide-react";
import { useParams } from "next/navigation";
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
import { useCreateProjectApiKey } from "@/hooks/api_keys/use-create-project-api-key";
import { useDeleteProjectApiKey } from "@/hooks/api_keys/use-delete-project-api-key";
import { useProjectApiKeys } from "@/hooks/api_keys/use-project-api-keys";

export default function ApiKeysPage() {
	const params = useParams();
	const projectId = params.projectId as string;

	const [showCreateDialog, setShowCreateDialog] = useState(false);
	const [formData, setFormData] = useState({
		name: "",
		description: "",
	});

	const { data: apiKeys = [], isLoading, error } = useProjectApiKeys(projectId);
	const createApiKey = useCreateProjectApiKey();
	const deleteApiKey = useDeleteProjectApiKey();

	const resetForm = () => {
		setFormData({
			name: "",
			description: "",
		});
	};

	const handleCreateApiKey = () => {
		if (!formData.name) return;

		createApiKey.mutate(
			{
				name: formData.name,
				projectId,
				status: "active",
			},
			{
				onSuccess: () => {
					setShowCreateDialog(false);
					resetForm();
				},
			},
		);
	};

	const handleDeleteApiKey = (id: string) => {
		deleteApiKey.mutate({ id });
	};

	if (isLoading) {
		return (
			<div className="flex min-h-[400px] items-center justify-center">
				<div className="text-center">
					<div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
					<p className="text-muted-foreground">Loading API keys...</p>
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="flex min-h-[400px] items-center justify-center">
				<div className="text-center">
					<h3 className="mb-2 font-medium text-foreground text-lg">
						Failed to load API keys
					</h3>
					<p className="mb-4 text-muted-foreground">{error.message}</p>
				</div>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<h1 className="font-bold text-2xl text-foreground">API keys</h1>
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
									disabled={!formData.name || createApiKey.isPending}
									className="bg-primary hover:bg-primary/90"
								>
									{createApiKey.isPending ? "Creating..." : "Create secret key"}
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
									Expires
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Created By
								</th>
								<th className="px-6 py-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
									Status
								</th>
								<th className="px-6 py-3 text-right font-medium text-gray-500 text-xs uppercase tracking-wider dark:text-gray-400">
									Actions
								</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-border">
							{apiKeys.map((apiKey) => (
								<tr key={apiKey.id} className="hover:bg-muted/50">
									<td className="whitespace-nowrap px-6 py-4 font-medium text-card-foreground text-sm">
										{apiKey.name}
									</td>
									<td className="whitespace-nowrap px-6 py-4 font-mono text-muted-foreground text-sm">
										{apiKey.key_preview}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{new Date(apiKey.created_at).toLocaleDateString("en-US", {
											year: "numeric",
											month: "short",
											day: "numeric",
										})}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{apiKey.expires_at
											? new Date(apiKey.expires_at).toLocaleDateString(
													"en-US",
													{
														year: "numeric",
														month: "short",
														day: "numeric",
													},
												)
											: "Never"}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										{apiKey.user_id}
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-muted-foreground text-sm">
										<span
											className={`inline-flex items-center rounded-full px-2 py-1 font-medium text-xs ${
												apiKey.status === "active"
													? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
													: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
											}`}
										>
											{apiKey.status}
										</span>
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
												disabled={deleteApiKey.isPending}
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
