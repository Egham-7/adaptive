import { NextResponse } from "next/server";
import { db } from "@/server/db";

export async function GET() {
	const startTime = Date.now();

	// Check database connectivity
	const dbStatus = await db.$queryRaw`SELECT 1`
		.then(() => "ok" as const)
		.catch((error) => {
			console.error("Database health check failed:", error);
			return "unhealthy" as const;
		});

	const responseTime = Date.now() - startTime;
	const overallStatus = dbStatus === "ok" ? "healthy" : "degraded";

	const response = {
		status: overallStatus,
		timestamp: new Date().toISOString(),
		service: "adaptive-app",
		version: process.env.npm_package_version || "unknown",
		checks: {
			database: dbStatus,
		},
		uptime: process.uptime(),
		responseTime: `${responseTime}ms`,
	};

	const statusCode = overallStatus === "healthy" ? 200 : 503;

	return NextResponse.json(response, { status: statusCode });
}
