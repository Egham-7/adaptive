import {
	Body,
	Container,
	Head,
	Html,
	Preview,
	Section,
	Text,
	Link,
} from "@react-email/components";
import * as React from "react";

interface CustomerConfirmationEmailProps {
	ticketId: string;
	name: string;
	subject: string;
	categoryLabel: string;
	priorityLabel: string;
}

export const CustomerConfirmationEmail = ({
	ticketId,
	name,
	subject,
	categoryLabel,
	priorityLabel,
}: CustomerConfirmationEmailProps) => {
	return (
		<Html>
			<Head />
			<Preview>Support Ticket Confirmation - {ticketId}</Preview>
			<Body style={main}>
				<Container style={container}>
					<Section style={header}>
						<Text style={headerTitle}>Support Ticket Received</Text>
						<Text style={headerSubtitle}>We've received your support request</Text>
					</Section>

					<Section style={content}>
						<Text style={greeting}>Hi {name},</Text>

						<Text style={paragraph}>
							Thank you for contacting Adaptive support. We've received your
							support ticket and will respond within 24 hours.
						</Text>

						<Section style={ticketDetails}>
							<Text style={detailsTitle}>
								<strong>Ticket Details:</strong>
							</Text>
							<ul style={detailsList}>
								<li>
									<strong>Ticket ID:</strong> {ticketId}
								</li>
								<li>
									<strong>Subject:</strong> {subject}
								</li>
								<li>
									<strong>Priority:</strong> {priorityLabel}
								</li>
								<li>
									<strong>Category:</strong> {categoryLabel}
								</li>
							</ul>
						</Section>

						<Text style={paragraph}>
							Please reference ticket ID <strong>{ticketId}</strong> in any
							follow-up communications.
						</Text>
					</Section>

					<Section style={footer}>
						<Text style={signature}>
							Best regards,
							<br />
							The Adaptive Support Team
						</Text>
						<Text style={websiteLink}>
							<Link href="https://llmadaptive.uk" style={link}>
								llmadaptive.uk
							</Link>
						</Text>
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
	padding: "0",
	borderRadius: "8px",
	boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
	maxWidth: "600px",
	overflow: "hidden",
};

const header = {
	backgroundColor: "#d4af37",
	color: "#ffffff",
	padding: "20px",
	textAlign: "center" as const,
};

const headerTitle = {
	fontSize: "24px",
	fontWeight: "bold",
	margin: "0",
	color: "#ffffff",
};

const headerSubtitle = {
	fontSize: "16px",
	margin: "5px 0 0 0",
	color: "#ffffff",
	opacity: "0.9",
};

const content = {
	backgroundColor: "#ffffff",
	padding: "20px",
	border: "1px solid #e9ecef",
	borderTop: "none",
};

const greeting = {
	fontSize: "16px",
	margin: "0 0 16px 0",
	color: "#333333",
};

const paragraph = {
	fontSize: "16px",
	lineHeight: "24px",
	margin: "16px 0",
	color: "#333333",
};

const ticketDetails = {
	backgroundColor: "#f8f9fa",
	padding: "15px",
	borderRadius: "4px",
	margin: "20px 0",
	border: "1px solid #e9ecef",
};

const detailsTitle = {
	fontSize: "16px",
	margin: "0 0 10px 0",
	color: "#333333",
};

const detailsList = {
	margin: "10px 0 0 20px",
	padding: "0",
	fontSize: "14px",
	lineHeight: "20px",
	color: "#555555",
};

const footer = {
	marginTop: "30px",
	paddingTop: "20px",
	borderTop: "1px solid #e9ecef",
	fontSize: "14px",
	color: "#666666",
	padding: "20px",
};

const signature = {
	margin: "0 0 10px 0",
	fontSize: "14px",
	color: "#666666",
};

const websiteLink = {
	margin: "0",
	fontSize: "14px",
};

const link = {
	color: "#d4af37",
	textDecoration: "none",
};

export default CustomerConfirmationEmail;