import {
  Sparkles,
  FileText,
  Code,
  GraduationCap,
  MessageSquare,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils"; // Assuming you have a cn utility for class merging

interface EmptyStateProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

export function EmptyState({ onSendMessage, isLoading }: EmptyStateProps) {
  const handleSuggestedQuestion = (question: string) => {
    onSendMessage(question);
  };

  const suggestedQuestions = [
    "How does AI work?",
    "Are black holes real?",
    'How many Rs are in the word "strawberry"?',
  ];

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-5xl mx-auto text-center px-4 py-10 space-y-14">
      <div className="space-y-6">
        <div className="flex items-center justify-center w-20 h-20 mx-auto rounded-full bg-primary/10 shadow-inner">
          <MessageSquare className="w-10 h-10 text-primary" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-linear-to-r from-primary to-primary/60 bg-clip-text text-transparent">
          How can I help you?
        </h1>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 w-full max-w-3xl">
        {[
          {
            icon: <Sparkles className="h-5 w-5" />,
            label: "Create",
            question: "Help me create something new",
          },
          {
            icon: <FileText className="h-5 w-5" />,
            label: "Explore",
            question: "I want to explore a topic",
          },
          {
            icon: <Code className="h-5 w-5" />,
            label: "Code",
            question: "Help me with coding",
          },
          {
            icon: <GraduationCap className="h-5 w-5" />,
            label: "Learn",
            question: "Teach me something new",
          },
        ].map((item, index) => (
          <Button
            key={index}
            variant="outline"
            className={cn(
              "rounded-xl px-4 py-6 h-auto flex flex-col items-center justify-center gap-3 hover:bg-primary/5 hover:border-primary/30 transition-all",
              isLoading && "opacity-50 pointer-events-none",
            )}
            onClick={() => handleSuggestedQuestion(item.question)}
            disabled={isLoading}
          >
            <div className="p-2 rounded-full bg-primary/10">{item.icon}</div>
            <span>{item.label}</span>
          </Button>
        ))}
      </div>

      <div className="space-y-3 w-full max-w-3xl bg-muted/30 rounded-xl p-4 border border-border/50">
        <h2 className="text-lg font-medium text-muted-foreground mb-4">
          Popular questions
        </h2>
        {suggestedQuestions.map((question, index) => (
          <div
            key={index}
            className="bg-background border border-border/50 rounded-lg py-3 px-4 hover:bg-primary/5 hover:border-primary/30 cursor-pointer transition-all"
            onClick={() => handleSuggestedQuestion(question)}
          >
            <p className="text-lg">{question}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
