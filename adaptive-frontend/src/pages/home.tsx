"use client";

import type React from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  MessageSquarePlusIcon,
  Settings,
  Zap,
  Target,
  BrainCircuit,
} from "lucide-react";
import { useRouter } from "@tanstack/react-router";
import { useCreateConversation } from "@/lib/hooks/use-create-conversation";

export default function Home() {
  const router = useRouter();

  const { mutateAsync } = useCreateConversation();

  const handleCreateChat = async () => {
    const conversation = await mutateAsync("New Conversation");
    router.navigate({
      to: "/conversations/$conversationId",
      params: { conversationId: String(conversation.id) },
    });
  };

  return (
    <div className="flex flex-col items-center text-center">
      <div className="mb-8 flex items-center justify-center">
        <div className="relative">
          <div className="absolute -top-6 -right-6">
            <BrainCircuit className="h-12 w-12 text-primary animate-pulse" />
          </div>
          <div className="bg-primary/10 p-6 rounded-full">
            <Settings className="h-16 w-16 text-primary animate-spin-slow" />
          </div>
        </div>
      </div>

      <h1 className="text-4xl font-bold mb-4">Welcome to Adaptive</h1>
      <p className="text-xl text-muted-foreground mb-6 max-w-md">
        The intelligent chat platform that automatically selects the optimal AI
        model for your specific needs.
      </p>

      <div className="bg-primary/5 rounded-lg p-4 mb-10 max-w-md">
        <p className="text-sm font-medium">
          Adaptive analyzes your prompts and automatically selects the best AI
          model based on your domain and requirements — no more guessing which
          model to use.
        </p>
      </div>

      <Button
        size="lg"
        className="text-lg px-8 py-6 h-auto gap-2 mb-12"
        onClick={handleCreateChat}
      >
        <MessageSquarePlusIcon className="h-5 w-5" />
        Start New Conversation
      </Button>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
        <FeatureCard
          icon={<Target className="h-8 w-8" />}
          title="Domain-Specific"
          description="Automatically detects your domain and selects specialized models for better results."
        />
        <FeatureCard
          icon={<Zap className="h-8 w-8" />}
          title="Intelligent Switching"
          description="Seamlessly switches between models based on the complexity of your prompts."
        />
        <FeatureCard
          icon={<BrainCircuit className="h-8 w-8" />}
          title="Optimized Performance"
          description="Get the best performance and accuracy without manually selecting models."
        />
      </div>
    </div>
  );
}

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="mb-2 text-primary">{icon}</div>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <CardDescription className="text-sm">{description}</CardDescription>
      </CardContent>
    </Card>
  );
}
