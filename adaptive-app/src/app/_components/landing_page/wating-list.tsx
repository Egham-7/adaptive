import { Globe } from "./globe";

export function WaitingListSection() {
  return (
    <div className="relative flexmax-w-lg items-center justify-center overflow-hidden rounded-lg border bg-background md:pb-60 md:shadow-xl">
      <div className="h-[20rem] w-full rounded-md bg-background relative flex flex-col items-center justify-center antialiased">
        <Globe />
        <div className="max-w-2xl mx-auto p-6 rounded-lg backdrop-blur-md bg-gradient-to-br from-white/90 via-white/80 to-white/70 dark:from-white/95 dark:via-white/90 dark:to-white/85 shadow-lg relative z-10">
          <h1 className="relative z-10 text-lg md:text-7xl bg-clip-text text-transparent bg-gradient-to-b from-gray-900 to-gray-600 dark:from-gray-900 dark:to-gray-700 text-center font-sans font-bold">
            Join the waitlist
          </h1>

          <p className="text-gray-700 dark:text-gray-800 max-w-lg mx-auto my-2 text-sm text-center relative z-10">
            Help us launch this software as soon as possible!
          </p>
          <Input
            type="email"
            placeholder="hi@manuarora.in"
            className="w-full mt-4 relative z-10 bg-white dark:bg-white"
          />
        </div>
      </div>
    </div>
  );
}

import React from "react";
import { Input } from "@/components/ui/input";
