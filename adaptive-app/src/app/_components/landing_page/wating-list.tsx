"use client"
import React, { useState } from "react";
import { BackgroundBeams } from "./background-beams";
import { Globe } from "./globe";
import { Input } from "@/components/ui/input";
import validator from "validator";

export function WaitingListSection() {
	const [email, setEmail] = useState("");
	const [status, setStatus] = useState<"idle" | "loading" | "success" | "duplicate" | "error">("idle");
	const [message, setMessage] = useState("");

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		setStatus("loading");
		setMessage("");
		try {
			if (!email || !validator.isEmail(email)) {
				setStatus("error");
				setMessage("Invalid email");
				return;
			}
			const res = await fetch("/api/waitlist", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ email }),
			});
			const data = await res.json();
			if (res.ok) {
				setStatus("success");
				setMessage("You're on the waitlist! 🎉");
				setEmail("");
				// Send welcome email (fire and forget)
				fetch("/api/waitlist/send-email", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ email }),
				});
			} else if (res.status === 409) {
				setStatus("duplicate");
				setMessage("This email is already on the waitlist.");
			} else {
				setStatus("error");
				setMessage(data.error || "Something went wrong.");
			}
		} catch (err) {
			setStatus("error");
			setMessage("Network error. Please try again.");
		}
	};

	return (
		<div className="flexmax-w-lg relative items-center justify-center overflow-hidden rounded-lg border bg-background md:pb-60 md:shadow-xl">
			<div className="relative flex h-[20rem] w-full flex-col items-center justify-center rounded-md bg-background antialiased">
				<Globe />
				<div className="relative z-10 mx-auto max-w-2xl rounded-lg bg-gradient-to-br from-white/90 via-white/80 to-white/70 p-6 shadow-lg backdrop-blur-md dark:from-white/95 dark:via-white/90 dark:to-white/85">
					<h1 className="relative z-10 bg-gradient-to-b from-gray-900 to-gray-600 bg-clip-text text-center font-bold font-sans text-lg text-transparent md:text-7xl dark:from-gray-900 dark:to-gray-700">
						Join the waitlist
					</h1>

					<p className="relative z-10 mx-auto my-2 max-w-lg text-center text-gray-700 text-sm dark:text-gray-800">
						Help us launch this software as soon as possible!
					</p>
					<form onSubmit={handleSubmit} className="w-full">
						<Input
							type="email"
							placeholder="hello@llmadaptive.uk"
							className="relative z-10 mt-4 w-full bg-white dark:bg-white"
							value={email}
							onChange={(e) => setEmail(e.target.value)}
							required
							disabled={status === "loading"}
						/>
						<button
							type="submit"
							className="mt-2 w-full rounded bg-black text-white py-2 font-semibold disabled:opacity-50"
							disabled={status === "loading"}
						>
							{status === "loading" ? "Joining..." : "Join Waitlist"}
						</button>
					</form>
					{message && (
						<div className="mt-2 text-center text-sm">
							{status === "success" ? (
								<span className="text-green-600">{message}</span>
							) : status === "duplicate" ? (
								<span className="text-yellow-600">{message}</span>
							) : status === "error" ? (
								<span className="text-red-600">{message}</span>
							) : null}
						</div>
					)}
				</div>
				<BackgroundBeams />
			</div>
		</div>
	);
}
