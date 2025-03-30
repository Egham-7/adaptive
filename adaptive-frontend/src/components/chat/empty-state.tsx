import {
  Sparkles,
  FileText,
  Code,
  GraduationCap,
  MessageSquare,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface EmptyStateProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

export function EmptyState({ onSendMessage, isLoading }: EmptyStateProps) {
  const handleSuggestedQuestion = (question: string) => {
    onSendMessage(question);
  };

  return (
    <div className="flex flex-col items-center justify-center w-full text-center p-6 space-y-12">
      <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
        <MessageSquare className="w-8 h-8 text-primary" />
      </div>

      <h1 className="text-4xl font-bold tracking-tight">How can I help you?</h1>

      <div className="flex flex-wrap gap-4 justify-center">
        <Button
          variant="outline"
          className="rounded-full px-6 py-6 h-auto"
          onClick={() =>
            handleSuggestedQuestion("Help me create something new")
          }
          disabled={isLoading}
        >
          <Sparkles className="mr-2 h-5 w-5" />
          Create
        </Button>

        <Button
          variant="outline"
          className="rounded-full px-6 py-6 h-auto"
          onClick={() => handleSuggestedQuestion("I want to explore a topic")}
          disabled={isLoading}
        >
          <FileText className="mr-2 h-5 w-5" />
          Explore
        </Button>

        <Button
          variant="outline"
          className="rounded-full px-6 py-6 h-auto"
          onClick={() => handleSuggestedQuestion("Help me with coding")}
          disabled={isLoading}
        >
          <Code className="mr-2 h-5 w-5" />
          Code
        </Button>

        <Button
          variant="outline"
          className="rounded-full px-6 py-6 h-auto"
          onClick={() => handleSuggestedQuestion("Teach me something new")}
          disabled={isLoading}
        >
          <GraduationCap className="mr-2 h-5 w-5" />
          Learn
        </Button>
      </div>

      <div className="space-y-4 w-full max-w-3xl">
        <div
          className="border-b border-border py-4 hover:bg-muted cursor-pointer transition-colors px-4 rounded-md"
          onClick={() => handleSuggestedQuestion("How does AI work?")}
        >
          <p className="text-lg">How does AI work?</p>
        </div>

        <div
          className="border-b border-border py-4 hover:bg-muted cursor-pointer transition-colors px-4 rounded-md"
          onClick={() => handleSuggestedQuestion("Are black holes real?")}
        >
          <p className="text-lg">Are black holes real?</p>
        </div>

        <div
          className="border-b border-border py-4 hover:bg-muted cursor-pointer transition-colors px-4 rounded-md"
          onClick={() =>
            handleSuggestedQuestion('How many Rs are in the word "strawberry"?')
          }
        >
          <p className="text-lg">How many Rs are in the word "strawberry"?</p>
        </div>
      </div>
    </div>
  );
}
