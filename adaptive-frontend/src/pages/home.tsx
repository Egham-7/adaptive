import {
  Plus,
  Cpu,
  ImageIcon,
  Send,
  FileText,
  GraduationCap,
  Eye,
  MoreHorizontal,
} from "lucide-react";
import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function Home() {
  const [message, setMessage] = useState("");
  const [selectedModel, setSelectedModel] = useState(undefined);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Submitted message:", message);
    setMessage("");
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-3xl space-y-8">
        {/* Logo and Heading */}
        <div className="flex flex-col items-center gap-4">
          <div className="relative w-16 h-16">
            <img src="https://v0.dev/placeholder.svg" className="rounded-lg" />
          </div>
          <h1 className="text-4xl font-display font-semibold text-center">
            Adaptive
          </h1>
        </div>

        {/* Model Selection */}
        <div className="flex items-center justify-center gap-2">
          <Cpu className="w-5 h-5 text-muted-foreground" />
          <Select defaultValue={selectedModel} disabled>
            <SelectTrigger className="w-[180px] bg-secondary border-0">
              <SelectValue placeholder="Selected Model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4">GPT-4 Turbo</SelectItem>
              <SelectItem value="gpt-3.5">GPT-3.5</SelectItem>
              <SelectItem value="claude">Claude 2.1</SelectItem>
            </SelectContent>
          </Select>
          <div className="text-xs text-muted-foreground">
            Auto-selected based on query
          </div>
        </div>

        {/* Message Input Area */}
        <form onSubmit={handleSubmit} className="relative">
          <div className="rounded-lg bg-card p-2">
            <div className="flex items-center gap-2 p-2">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="rounded-md"
              >
                <Plus className="h-5 w-5" />
              </Button>
              <Input
                type="text"
                placeholder="Message Adaptive..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="flex-1 bg-transparent border-none focus:outline-none focus:ring-0 placeholder:text-muted-foreground"
              />
              <Button
                type="submit"
                variant="ghost"
                size="icon"
                className="rounded-md"
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </form>

        {/* Action Buttons */}
        <div className="flex flex-wrap justify-center gap-2">
          <Button variant="secondary" className="rounded-md gap-2">
            <ImageIcon className="h-4 w-4" />
            Create image
          </Button>
          <Button variant="secondary" className="rounded-md gap-2">
            <FileText className="h-4 w-4" />
            Summarize text
          </Button>
          <Button variant="secondary" className="rounded-md gap-2">
            <Eye className="h-4 w-4" />
            Analyze images
          </Button>
          <Button variant="secondary" className="rounded-md gap-2">
            <GraduationCap className="h-4 w-4" />
            Get advice
          </Button>
          <Button variant="secondary" className="rounded-md">
            <MoreHorizontal className="h-4 w-4" />
            More
          </Button>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-muted-foreground">
          ChatGPT can make mistakes. Check important info.
        </p>
      </div>
    </div>
  );
}
