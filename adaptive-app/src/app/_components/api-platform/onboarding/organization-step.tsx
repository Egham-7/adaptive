import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowLeft, Building2, ChevronRight } from "lucide-react";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	Form,
	FormControl,
	FormDescription,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const organizationSchema = z.object({
	name: z.string().min(1, "Organization name is required"),
	description: z.string().optional(),
});

interface OrganizationStepProps {
	onSubmit: (values: z.infer<typeof organizationSchema>) => void;
	onBack: () => void;
	isLoading: boolean;
}

export function OrganizationStep({
	onSubmit,
	onBack,
	isLoading,
}: OrganizationStepProps) {
	const form = useForm<z.infer<typeof organizationSchema>>({
		resolver: zodResolver(organizationSchema),
		defaultValues: {
			name: "",
			description: "",
		},
	});

	return (
		<Card className="border-2">
			<CardHeader>
				<div className="mb-4 flex items-center gap-2">
					<Button variant="ghost" size="sm" onClick={onBack} className="p-2">
						<ArrowLeft className="h-4 w-4" />
					</Button>
					<div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
						<Building2 className="h-5 w-5 text-primary" />
					</div>
				</div>
				<CardTitle className="text-xl">Create your organization</CardTitle>
				<CardDescription>
					This will be your main workspace for managing AI projects and team
					collaboration.
				</CardDescription>
			</CardHeader>
			<CardContent>
				<Form {...form}>
					<form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
						<FormField
							control={form.control}
							name="name"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Organization Name</FormLabel>
									<FormControl>
										<Input placeholder="e.g., Acme AI Solutions" {...field} />
									</FormControl>
									<FormDescription>
										Choose a name that represents your company or team
									</FormDescription>
									<FormMessage />
								</FormItem>
							)}
						/>
						<FormField
							control={form.control}
							name="description"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Description (optional)</FormLabel>
									<FormControl>
										<Textarea
											placeholder="What does your organization do?"
											rows={3}
											{...field}
										/>
									</FormControl>
									<FormDescription>
										Help your team understand the purpose of this organization
									</FormDescription>
									<FormMessage />
								</FormItem>
							)}
						/>
						<div className="flex justify-end gap-3">
							<Button type="button" variant="outline" onClick={onBack}>
								Back
							</Button>
							<Button type="submit" disabled={isLoading} className="min-w-32">
								{isLoading ? "Creating..." : "Continue"}
								<ChevronRight className="ml-2 h-4 w-4" />
							</Button>
						</div>
					</form>
				</Form>
			</CardContent>
		</Card>
	);
}
