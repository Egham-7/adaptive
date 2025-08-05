"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2 } from "lucide-react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";
import { Button } from "@/components/ui/button";
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
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/trpc/react";

const supportTicketSchema = z.object({
	name: z.string().min(2, "Name must be at least 2 characters"),
	email: z.string().email("Please enter a valid email address"),
	category: z.enum([
		"technical",
		"billing",
		"feature-request",
		"bug-report",
		"general",
	]),
	priority: z.enum(["low", "medium", "high", "urgent"]),
	subject: z.string().min(5, "Subject must be at least 5 characters"),
	description: z.string().min(20, "Description must be at least 20 characters"),
});

type SupportTicketForm = z.infer<typeof supportTicketSchema>;

const categoryLabels = {
	technical: "Technical Support",
	billing: "Billing & Payments",
	"feature-request": "Feature Request",
	"bug-report": "Bug Report",
	general: "General Inquiry",
};

const priorityLabels = {
	low: "Low",
	medium: "Medium",
	high: "High",
	urgent: "Urgent",
};

export default function SupportForm() {
	const form = useForm<SupportTicketForm>({
		resolver: zodResolver(supportTicketSchema),
		defaultValues: {
			name: "",
			email: "",
			category: "general",
			priority: "medium",
			subject: "",
			description: "",
		},
	});

	const submitTicket = api.support.submitTicket.useMutation({
		onSuccess: (data) => {
			toast.success(`Support ticket submitted! Ticket ID: ${data.ticketId}`);
			form.reset();
		},
		onError: (error) => {
			toast.error(
				error.message || "Failed to submit ticket. Please try again.",
			);
		},
	});

	const onSubmit = (data: SupportTicketForm) => {
		submitTicket.mutate(data);
	};

	return (
		<Form {...form}>
			<form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
				<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
					<FormField
						control={form.control}
						name="name"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Full Name</FormLabel>
								<FormControl>
									<Input
										placeholder="John Doe"
										{...field}
										disabled={submitTicket.isPending}
									/>
								</FormControl>
								<FormMessage />
							</FormItem>
						)}
					/>

					<FormField
						control={form.control}
						name="email"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Email Address</FormLabel>
								<FormControl>
									<Input
										type="email"
										placeholder="john@example.com"
										{...field}
										disabled={submitTicket.isPending}
									/>
								</FormControl>
								<FormMessage />
							</FormItem>
						)}
					/>
				</div>

				<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
					<FormField
						control={form.control}
						name="category"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Category</FormLabel>
								<Select
									onValueChange={field.onChange}
									defaultValue={field.value}
									disabled={submitTicket.isPending}
								>
									<FormControl>
										<SelectTrigger>
											<SelectValue placeholder="Select a category" />
										</SelectTrigger>
									</FormControl>
									<SelectContent>
										{Object.entries(categoryLabels).map(([value, label]) => (
											<SelectItem key={value} value={value}>
												{label}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
								<FormMessage />
							</FormItem>
						)}
					/>

					<FormField
						control={form.control}
						name="priority"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Priority</FormLabel>
								<Select
									onValueChange={field.onChange}
									defaultValue={field.value}
									disabled={submitTicket.isPending}
								>
									<FormControl>
										<SelectTrigger>
											<SelectValue placeholder="Select priority" />
										</SelectTrigger>
									</FormControl>
									<SelectContent>
										{Object.entries(priorityLabels).map(([value, label]) => (
											<SelectItem key={value} value={value}>
												{label}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
								<FormMessage />
							</FormItem>
						)}
					/>
				</div>

				<FormField
					control={form.control}
					name="subject"
					render={({ field }) => (
						<FormItem>
							<FormLabel>Subject</FormLabel>
							<FormControl>
								<Input
									placeholder="Brief description of your issue"
									{...field}
									disabled={submitTicket.isPending}
								/>
							</FormControl>
							<FormMessage />
						</FormItem>
					)}
				/>

				<FormField
					control={form.control}
					name="description"
					render={({ field }) => (
						<FormItem>
							<FormLabel>Description</FormLabel>
							<FormControl>
								<Textarea
									placeholder="Please provide detailed information about your issue, including any error messages, steps to reproduce, or relevant context..."
									rows={6}
									{...field}
									disabled={submitTicket.isPending}
								/>
							</FormControl>
							<FormDescription>
								Please be as detailed as possible to help us resolve your issue
								quickly.
							</FormDescription>
							<FormMessage />
						</FormItem>
					)}
				/>

				<Button
					type="submit"
					disabled={submitTicket.isPending}
					className="w-full"
				>
					{submitTicket.isPending ? (
						<>
							<Loader2 className="mr-2 h-4 w-4 animate-spin" />
							Submitting Ticket...
						</>
					) : (
						"Submit Support Ticket"
					)}
				</Button>

				<p className="text-muted-foreground text-sm">
					We typically respond within 24 hours. For urgent issues, please mark
					the priority as "Urgent".
				</p>
			</form>
		</Form>
	);
}
