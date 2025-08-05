import { TRPCError } from "@trpc/server";
import { Resend } from "resend";
import { z } from "zod";
import { env } from "@/env";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";

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

				// Email to support team
				const supportEmailHtml = `
					<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
						<div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
							<h2 style="color: #333; margin: 0;">New Support Ticket</h2>
							<p style="color: #666; margin: 5px 0 0 0;">Ticket ID: <strong>${ticketId}</strong></p>
						</div>
						
						<div style="background-color: white; padding: 20px; border: 1px solid #e9ecef; border-radius: 8px;">
							<table style="width: 100%; border-collapse: collapse;">
								<tr>
									<td style="padding: 8px 0; font-weight: bold; width: 120px;">Name:</td>
									<td style="padding: 8px 0;">${input.name}</td>
								</tr>
								<tr>
									<td style="padding: 8px 0; font-weight: bold;">Email:</td>
									<td style="padding: 8px 0;">${input.email}</td>
								</tr>
								<tr>
									<td style="padding: 8px 0; font-weight: bold;">Category:</td>
									<td style="padding: 8px 0;">${categoryLabel}</td>
								</tr>
								<tr>
									<td style="padding: 8px 0; font-weight: bold;">Priority:</td>
									<td style="padding: 8px 0;">
										<span style="background-color: ${input.priority === "urgent" ? "#dc3545" : input.priority === "high" ? "#fd7e14" : input.priority === "medium" ? "#ffc107" : "#28a745"}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
											${priorityLabel}
										</span>
									</td>
								</tr>
								<tr>
									<td style="padding: 8px 0; font-weight: bold;">Subject:</td>
									<td style="padding: 8px 0;">${input.subject}</td>
								</tr>
							</table>
							
							<div style="margin-top: 20px;">
								<h4 style="color: #333; margin-bottom: 10px;">Description:</h4>
								<div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; border-left: 4px solid #007bff;">
									${input.description.replace(/\n/g, "<br>")}
								</div>
							</div>
						</div>
						
						<div style="margin-top: 20px; padding: 15px; background-color: #e7f3ff; border-radius: 4px; font-size: 14px; color: #666;">
							<p style="margin: 0;"><strong>Next Steps:</strong></p>
							<ul style="margin: 10px 0 0 20px; padding: 0;">
								<li>Reply to this email to respond to the customer</li>
								<li>Customer will receive a copy of your response</li>
								<li>Reference ticket ID: ${ticketId} in all communications</li>
							</ul>
						</div>
					</div>
				`;

				// Confirmation email to customer
				const customerEmailHtml = `
					<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
						<div style="background-color: #007bff; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
							<h2 style="margin: 0;">Support Ticket Received</h2>
							<p style="margin: 5px 0 0 0; opacity: 0.9;">We've received your support request</p>
						</div>
						
						<div style="background-color: white; padding: 20px; border: 1px solid #e9ecef; border-top: none; border-radius: 0 0 8px 8px;">
							<p>Hi ${input.name},</p>
							
							<p>Thank you for contacting Adaptive support. We've received your support ticket and will respond within 24 hours.</p>
							
							<div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0;">
								<p style="margin: 0;"><strong>Ticket Details:</strong></p>
								<ul style="margin: 10px 0 0 20px; padding: 0;">
									<li><strong>Ticket ID:</strong> ${ticketId}</li>
									<li><strong>Subject:</strong> ${input.subject}</li>
									<li><strong>Priority:</strong> ${priorityLabel}</li>
									<li><strong>Category:</strong> ${categoryLabel}</li>
								</ul>
							</div>
							
							<p>Please reference ticket ID <strong>${ticketId}</strong> in any follow-up communications.</p>
							
							<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef; font-size: 14px; color: #666;">
								<p>Best regards,<br>The Adaptive Support Team</p>
								<p><a href="https://llmadaptive.uk" style="color: #007bff;">llmadaptive.uk</a></p>
							</div>
						</div>
					</div>
				`;

				// Send email to support team
				await resend.emails.send({
					from: "Adaptive Support <support@llmadaptive.uk>",
					to: ["support@llmadaptive.uk"], // Update with your actual support email
					subject: `[${priorityLabel}] ${categoryLabel}: ${input.subject} (${ticketId})`,
					html: supportEmailHtml,
					replyTo: input.email,
				});

				// Send confirmation email to customer
				await resend.emails.send({
					from: "Adaptive Support <support@llmadaptive.uk>",
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
