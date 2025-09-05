import { zodResolver } from "@hookform/resolvers/zod";
import { Pencil } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import {
	Drawer,
	DrawerContent,
	DrawerDescription,
	DrawerHeader,
	DrawerTitle,
	DrawerTrigger,
} from "@/components/ui/drawer";
import {
	Form,
	FormControl,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { useUpdateConversation } from "@/hooks/conversations/use-update-conversation";
import { useIsMobile } from "@/hooks/use-mobile";
import type { ConversationListItem } from "@/types";

const formSchema = z.object({
	title: z.string().min(1, "Title is required"),
});

type FormValues = z.infer<typeof formSchema>;

// --- Child Form Component ---

interface EditConversationFormProps {
	conversation: ConversationListItem;
	onSubmit: (values: FormValues) => void;
	onCancel: () => void;
	isPending: boolean;
}

function EditConversationForm({
	conversation,
	onSubmit,
	onCancel,
	isPending,
}: EditConversationFormProps) {
	const form = useForm<FormValues>({
		resolver: zodResolver(formSchema),
		defaultValues: {
			title: conversation.title,
		},
	});

	return (
		<Form {...form}>
			<form onSubmit={form.handleSubmit(onSubmit)}>
				<div className="space-y-4">
					<FormField
						control={form.control}
						name="title"
						render={({ field }) => (
							<FormItem>
								<FormLabel>Conversation Title</FormLabel>
								<FormControl>
									<Input placeholder="Enter a title" {...field} />
								</FormControl>
								<FormMessage />
							</FormItem>
						)}
					/>
				</div>
				<div className="mt-4 flex justify-end gap-2">
					<Button type="button" variant="outline" onClick={onCancel}>
						Cancel
					</Button>
					<Button
						type="submit"
						disabled={isPending}
						onClick={() => form.handleSubmit(onSubmit)()}
					>
						{isPending ? "Saving..." : "Save Changes"}
					</Button>
				</div>
			</form>
		</Form>
	);
}

// --- Main Dialog/Drawer Component ---

interface EditConversationDialogProps {
	conversation: ConversationListItem;
}

export function EditConversationDialog({
	conversation,
}: EditConversationDialogProps) {
	const isMobile = useIsMobile();
	const [isOpen, setIsOpen] = useState(false);
	const updateConversationMutation = useUpdateConversation();

	const handleSubmit = (values: FormValues) => {
		updateConversationMutation.mutate(
			{
				id: conversation.id,
				title: values.title,
			},
			{
				onSuccess: () => {
					setIsOpen(false); // Close dialog on success
				},
			},
		);
	};

	const EditTrigger = (
		<Tooltip>
			<TooltipTrigger asChild>
				<Button
					variant="ghost"
					size="icon"
					className="h-7 w-7"
					disabled={updateConversationMutation.isPending}
				>
					<Pencil className="h-4 w-4" />
					<span className="sr-only">Edit</span>
				</Button>
			</TooltipTrigger>
			<TooltipContent side="bottom">Edit conversation</TooltipContent>
		</Tooltip>
	);

	if (isMobile) {
		return (
			<Drawer open={isOpen} onOpenChange={setIsOpen}>
				<DrawerTrigger asChild>{EditTrigger}</DrawerTrigger>
				<DrawerContent>
					<DrawerHeader className="text-left">
						<DrawerTitle>Edit Conversation</DrawerTitle>
						<DrawerDescription>
							Change the title of your conversation.
						</DrawerDescription>
					</DrawerHeader>
					<div className="px-4 pb-4">
						<EditConversationForm
							conversation={conversation}
							onSubmit={handleSubmit}
							onCancel={() => setIsOpen(false)}
							isPending={updateConversationMutation.isPending}
						/>
					</div>
				</DrawerContent>
			</Drawer>
		);
	}

	return (
		<Dialog open={isOpen} onOpenChange={setIsOpen}>
			<DialogTrigger asChild>{EditTrigger}</DialogTrigger>
			<DialogContent>
				<DialogHeader>
					<DialogTitle>Edit Conversation</DialogTitle>
					<DialogDescription>
						Change the title of your conversation.
					</DialogDescription>
				</DialogHeader>
				<EditConversationForm
					conversation={conversation}
					onSubmit={handleSubmit}
					onCancel={() => setIsOpen(false)}
					isPending={updateConversationMutation.isPending}
				/>
			</DialogContent>
		</Dialog>
	);
}
