import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(req: NextRequest) {
  try {
    const { email } = await req.json();
    if (!email || typeof email !== 'string') {
      return NextResponse.json({ error: 'Invalid email' }, { status: 400 });
    }

    // Save to database
    const waitlistEntry = await prisma.waitlistEmail.create({
      data: { email },
    });

    return NextResponse.json({ success: true, entry: waitlistEntry });
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Unique constraint failed
      return NextResponse.json({ error: 'Email already on waitlist' }, { status: 409 });
    }
    return NextResponse.json({ error: error.message || 'Internal server error' }, { status: 500 });
  }
} 