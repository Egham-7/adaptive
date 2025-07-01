// adaptive-app/src/app/api/waitlist/send-email/route.ts
import { NextRequest, NextResponse } from "next/server";
import { Resend } from "resend";

const resend = new Resend(process.env.RESEND_API_KEY);

export async function POST(req: NextRequest) {
  const { email } = await req.json();

  try {
    const data = await resend.emails.send({
      from: "Your Name <onboarding@resend.dev>", // Use your verified sender
      to: [email],
      subject: "Welcome to the Waitlist!",
      html: "<strong>Thanks for joining the waitlist!</strong>",
    });

    return NextResponse.json({ success: true, data });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}