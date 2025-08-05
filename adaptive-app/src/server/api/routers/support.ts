import { TRPCError } from "@trpc/server";
import { Resend } from "resend";
import { z } from "zod";
import { render } from "@react-email/render";
import { env } from "@/env";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import { SupportTicketEmail } from "@/components/emails/support-ticket-email";
import { CustomerConfirmationEmail } from "@/components/emails/customer-confirmation-email";

const resend = new Resend(env.RESEND_API_KEY);

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

export const supportRouter = createTRPCRouter({
	submitTicket: publicProcedure
		.input(supportTicketSchema)
		.mutation(async ({ input }) => {
			try {
				// Generate a unique ticket ID
				const ticketId = `TKT-${Date.now()}-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;

				const categoryLabel = categoryLabels[input.category];
				const priorityLabel = priorityLabels[input.priority];

				// Render support ticket email using React component
				const supportEmailHtml = await render(
					<SupportTicketEmail
						ticketId={ticketId}
						name={input.name}
						email={input.email}
						category={input.category}
						categoryLabel={categoryLabel}
						priority={input.priority}
						priorityLabel={priorityLabel}
						subject={input.subject}
						description={input.description}
					/>
				);

				// Render customer confirmation email using React component
				const customerEmailHtml = await render(
					<CustomerConfirmationEmail
						ticketId={ticketId}
						name={input.name}
						subject={input.subject}
						categoryLabel={categoryLabel}
						priorityLabel={priorityLabel}
					/>
				);

				// Send email to support team
				await resend.emails.send({
					from: "Adaptive Support <info@llmadaptive.uk>",
					to: ["info@llmadaptive.uk"],
					subject: `[${priorityLabel}] ${categoryLabel}: ${input.subject} (${ticketId})`,
					html: supportEmailHtml,
					replyTo: input.email,
				});

				// Send confirmation email to customer
				await resend.emails.send({
					from: "Adaptive Support <info@llmadaptive.uk>",
					to: [input.email],
					subject: `Support Ticket Confirmation - ${ticketId}`,
					html: customerEmailHtml,
				});

				return {
					success: true,
					ticketId,
					message: "Support ticket submitted successfully",
				};
			} catch (error) {
				console.error("Error sending support email:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to submit support ticket. Please try again.",
				});
			}
		}),
});
