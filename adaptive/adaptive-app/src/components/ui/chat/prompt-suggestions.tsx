import { AnimatePresence, motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { BookOpen, Code, PenTool } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

export interface PromptCategory {
  id: string;
  label: string;
  icon: LucideIcon;
  title: string;
  suggestions: string[];
}

interface PromptSuggestionsProps {
  label: string;
  onSuggestionClick: (text: string) => void;
  suggestions: string[];
  enableCategories?: boolean;
  categories?: PromptCategory[];
}

// Default categories for backward compatibility
const defaultCategories: PromptCategory[] = [
  {
    id: "learn",
    label: "Learn",
    icon: BookOpen,
    title: "Learning suggestions",
    suggestions: [
      "Explain the Big Bang theory",
      "How does photosynthesis work?",
      "What are black holes?",
      "Explain quantum computing",
      "How does the human brain work?",
    ],
  },
  {
    id: "code",
    label: "Code",
    icon: Code,
    title: "Coding suggestions",
    suggestions: [
      "Create a React component for a todo list",
      "Write a Python function to sort a list",
      "How to implement authentication in Next.js",
      "Explain async/await in JavaScript",
      "Create a CSS animation for a button",
    ],
  },
  {
    id: "write",
    label: "Write",
    icon: PenTool,
    title: "Writing suggestions",
    suggestions: [
      "Write a professional email to a client",
      "Create a product description for a smartphone",
      "Draft a blog post about AI",
      "Write a creative story about space exploration",
      "Create a social media post about sustainability",
    ],
  },
];

export function PromptSuggestions({
  label,
  onSuggestionClick,
  suggestions,
  enableCategories = false,
  categories = defaultCategories,
}: PromptSuggestionsProps) {
  const [activeCommandCategory, setActiveCommandCategory] = useState<
    string | null
  >(null);

  const handleCommandSelect = (command: string) => {
    onSuggestionClick(command);
    // Don't close the category immediately - let the parent component handle hiding suggestions
  };

  if (enableCategories) {
    return (
      <div className="w-full space-y-6">
        <motion.div
          className="text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <motion.div
            className="flex flex-col justify-center items-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 1 }}
          >
            <motion.h1
              className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60 mb-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.2 }}
            >
              Ready to assist you
            </motion.h1>
            <motion.p
              className="text-muted-foreground max-w-md"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.4 }}
            >
              Ask me anything or try one of the suggestions below
            </motion.p>
          </motion.div>
        </motion.div>

        {/* Command categories */}
        <motion.div
          className={cn(
            "w-full grid gap-4 mb-4",
            categories.length <= 3
              ? "grid-cols-3"
              : "grid-cols-2 md:grid-cols-3 lg:grid-cols-4",
          )}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.6 }}
        >
          {categories.map((category, index) => {
            const Icon = category.icon;
            return (
              <motion.div
                key={category.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.8 + index * 0.2 }}
              >
                <CommandButton
                  icon={<Icon className="w-5 h-5" />}
                  label={category.label}
                  isActive={activeCommandCategory === category.id}
                  onClick={() =>
                    setActiveCommandCategory(
                      activeCommandCategory === category.id
                        ? null
                        : category.id,
                    )
                  }
                />
              </motion.div>
            );
          })}
        </motion.div>

        {/* Command suggestions */}
        <AnimatePresence>
          {activeCommandCategory &&
            (() => {
              const activeCategory = categories.find(
                (cat) => cat.id === activeCommandCategory,
              );
              if (!activeCategory) return null;

              const Icon = activeCategory.icon;

              return (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="w-full overflow-hidden"
                >
                  <div className="bg-background rounded-xl border border-border shadow-sm overflow-hidden">
                    <div className="p-3 border-b border-border">
                      <h3 className="text-sm font-medium text-foreground">
                        {activeCategory.title}
                      </h3>
                    </div>
                    <ul className="divide-y divide-border">
                      {activeCategory.suggestions.map((suggestion, index) => (
                        <motion.li
                          key={index}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: index * 0.03 }}
                          onClick={() => handleCommandSelect(suggestion)}
                          className="p-3 hover:bg-muted cursor-pointer transition-colors duration-75"
                        >
                          <div className="flex items-center gap-3">
                            <Icon className="w-4 h-4 text-primary" />
                            <span className="text-sm text-foreground">
                              {suggestion}
                            </span>
                          </div>
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              );
            })()}
        </AnimatePresence>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-center font-bold text-2xl">{label}</h2>
      <div className="flex gap-6 text-sm">
        {suggestions.map((suggestion) => (
          <button
            type="button"
            key={suggestion}
            onClick={() => onSuggestionClick(suggestion)}
            className="h-max flex-1 rounded-xl border bg-background p-4 hover:bg-muted"
          >
            <p>{suggestion}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

interface CommandButtonProps {
  icon: React.ReactNode;
  label: string;
  isActive: boolean;
  onClick: () => void;
}

function CommandButton({ icon, label, isActive, onClick }: CommandButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      className={cn(
        "flex flex-col items-center w-full justify-center gap-2 p-4 rounded-xl border transition-all",
        isActive
          ? "bg-primary/10 border-primary/20 shadow-sm"
          : "bg-background border-border hover:border-border/60",
      )}
      whileHover={{
        scale: 1.05,
        boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      <div className={cn(isActive ? "text-primary" : "text-muted-foreground")}>
        {icon}
      </div>
      <span
        className={cn(
          "text-sm font-medium",
          isActive ? "text-primary" : "text-foreground",
        )}
      >
        {label}
      </span>
    </motion.button>
  );
}
