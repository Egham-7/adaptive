"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
  DialogTitle,
} from "@/components/ui/dialog";
import ChatbotPricing from "../../landing_page/chatbot-pricing";

interface PlansModalProps {
  children: React.ReactNode;
}

export function PlansModal({ children }: PlansModalProps) {
  const [open, setOpen] = useState(false);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="!max-w-2xl w-full">
        <DialogTitle>View Plans</DialogTitle>
        <ChatbotPricing />
      </DialogContent>
    </Dialog>
  );
}
