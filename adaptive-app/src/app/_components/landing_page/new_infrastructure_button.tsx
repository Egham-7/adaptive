"use client";
import { ArrowRight, Rocket } from "lucide-react";

import Link from "next/link";

export default function ApiButton() {
  return (
    <>
      <Link
        href="/"
        className="mx-auto flex w-fit items-center gap-2 rounded-md border p-1 pr-3"
      >
        <span className="rounded-sm bg-muted px-2 py-1 font-medium text-xs">
          New
        </span>
        <span className="font-medium text-sm">Adaptive Chat Platform</span>
        <span className="block h-4 w-px bg-border" />
        <ArrowRight className="size-4" />
      </Link>
    </>
  );
}
