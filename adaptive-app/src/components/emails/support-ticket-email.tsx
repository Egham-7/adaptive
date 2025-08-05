import {
	Body,
	Container,
	Head,
	Html,
	Preview,
	Section,
	Text,
	Hr,
} from "@react-email/components";
import * as React from "react";

interface SupportTicketEmailProps {
	ticketId: string;
	name: string;
	email: string;
	category: string;
	categoryLabel: string;
	priority: "low" | "medium" | "high" | "urgent";
	priorityLabel: string;
	subject: string;
	description: string;
}

export const SupportTicketEmail = ({
	ticketId,
	name,
	email,
	category,
	categoryLabel,
	priority,
	priorityLabel,
	subject,
	description,
}: SupportTicketEmailProps) => {
	const priorityColor = {
		urgent: "#dc3545",
		high: "#fd7e14",
		medium: "#ffc107",
		low: "#28a745",
	}[priority];

	return (
		<Html>
			<Head />
			<Preview>New Support Ticket - {ticketId}</Preview>
			<Body style={main}>
				<Container style={container}>
					<Section style={header}>
						<Text style={headerTitle}>New Support Ticket</Text>
						<Text style={headerSubtitle}>
							Ticket ID: <strong>{ticketId}</strong>
						</Text>
					</Section>

					<Section style={content}>
						<table style={table}>
							<tr>
								<td style={labelCell}>Name:</td>
								<td style={valueCell}>{name}</td>
							</tr>
							<tr>
								<td style={labelCell}>Email:</td>
								<td style={valueCell}>{email}</td>
							</tr>
							<tr>
								<td style={labelCell}>Category:</td>
								<td style={valueCell}>{categoryLabel}</td>
							</tr>
							<tr>
								<td style={labelCell}>Priority:</td>
								<td style={valueCell}>
									<span
										style={{
											...priorityBadge,
											backgroundColor: priorityColor,
										}}
									>
										{priorityLabel}
									</span>
								</td>
							</tr>
							<tr>
								<td style={labelCell}>Subject:</td>
								<td style={valueCell}>{subject}</td>
							</tr>
						</table>

						<Section style={descriptionSection}>
							<Text style={descriptionTitle}>Description:</Text>
							<Section style={descriptionBox}>
								<Text style={descriptionText}>
									{description.split('\n').map((line, index) => (
										<React.Fragment key={index}>
											{line}
											{index < description.split('\n').length - 1 && <br />}
										</React.Fragment>
									))}
								</Text>
							</Section>
						</Section>
					</Section>

					<Section style={footer}>
						<Text style={footerTitle}>
							<strong>Next Steps:</strong>
						</Text>
						<ul style={footerList}>
							<li>Reply to this email to respond to the customer</li>
							<li>Customer will receive a copy of your response</li>
							<li>Reference ticket ID: {ticketId} in all communications</li>
						</ul>
					</Section>
				</Container>
			</Body>
		</Html>
	);
};

const main = {
	backgroundColor: "#f6f9fc",
	fontFamily: '"Graphie", Arial, sans-serif',
	padding: "20px 0",
};

const container = {
	backgroundColor: "#ffffff",
	margin: "0 auto",
	padding: "20px",
	borderRadius: "8px",
	boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
	maxWidth: "600px",
};

const header = {
	backgroundColor: "#d4af37",
	color: "#ffffff",
	padding: "20px",
	borderRadius: "8px",
	marginBottom: "20px",
};

const headerTitle = {
	fontSize: "24px",
	fontWeight: "bold",
	margin: "0",
	color: "#ffffff",
};

const headerSubtitle = {
	fontSize: "14px",
	margin: "5px 0 0 0",
	color: "#ffffff",
	opacity: "0.9",
};

const content = {
	backgroundColor: "#ffffff",
	padding: "20px",
	border: "1px solid #e9ecef",
	borderRadius: "8px",
};

const table = {
	width: "100%",
	borderCollapse: "collapse" as const,
};

const labelCell = {
	padding: "8px 0",
	fontWeight: "bold",
	width: "120px",
	verticalAlign: "top" as const,
};

const valueCell = {
	padding: "8px 0",
	verticalAlign: "top" as const,
};

const priorityBadge = {
	color: "#ffffff",
	padding: "2px 8px",
	borderRadius: "4px",
	fontSize: "12px",
	fontWeight: "bold",
};

const descriptionSection = {
	marginTop: "20px",
};

const descriptionTitle = {
	fontSize: "16px",
	fontWeight: "bold",
	color: "#333333",
	marginBottom: "10px",
};

const descriptionBox = {
	backgroundColor: "#f8f9fa",
	padding: "15px",
	borderRadius: "4px",
	borderLeft: "4px solid #d4af37",
};

const descriptionText = {
	margin: "0",
	fontSize: "14px",
	lineHeight: "1.5",
	color: "#333333",
};

const footer = {
	marginTop: "20px",
	padding: "15px",
	backgroundColor: "#e7f3ff",
	borderRadius: "4px",
	fontSize: "14px",
	color: "#666666",
};

const footerTitle = {
	margin: "0 0 10px 0",
	fontSize: "14px",
};

const footerList = {
	margin: "10px 0 0 20px",
	padding: "0",
};

export default SupportTicketEmail;