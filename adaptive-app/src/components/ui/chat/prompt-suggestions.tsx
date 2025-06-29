import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, Code, PenTool } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface PromptSuggestionsProps {
  label: string;
  sendMessage: (message: { text: string }) => void;
  suggestions: string[];
  enableCategories?: boolean;
}

export function PromptSuggestions({
  label,
  sendMessage,
  suggestions,
  enableCategories = false,
}: PromptSuggestionsProps) {
  const [activeCommandCategory, setActiveCommandCategory] = useState<
    string | null
  >(null);

  const commandSuggestions = {
    learn: [
      "Explain the Big Bang theory",
      "How does photosynthesis work?",
      "What are black holes?",
      "Explain quantum computing",
      "How does the human brain work?",
    ],
    code: [
      "Create a React component for a todo list",
      "Write a Python function to sort a list",
      "How to implement authentication in Next.js",
      "Explain async/await in JavaScript",
      "Create a CSS animation for a button",
    ],
    write: [
      "Write a professional email to a client",
      "Create a product description for a smartphone",
      "Draft a blog post about AI",
      "Write a creative story about space exploration",
      "Create a social media post about sustainability",
    ],
  };

  const handleCommandSelect = (command: string) => {
    sendMessage({ text: command });
    setActiveCommandCategory(null);
  };

  if (enableCategories) {
    return (
      <div className="w-full space-y-6">
        {/* Logo with animated gradient */}
        <motion.div
          className="mb-8 w-20 h-20 relative mx-auto"
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{
            duration: 1,
            ease: "easeOut",
            delay: 0.2,
          }}
        >
          <motion.svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 200 200"
            width="100%"
            height="100%"
            className="w-full h-full"
            animate={{
              rotate: [0, 5, -5, 0],
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            <g clipPath="url(#cs_clip_1_ellipse-12)">
              <mask
                id="cs_mask_1_ellipse-12"
                style={{ maskType: "alpha" }}
                width="200"
                height="200"
                x="0"
                y="0"
                maskUnits="userSpaceOnUse"
              >
                <path
                  fill="#fff"
                  fillRule="evenodd"
                  d="M100 150c27.614 0 50-22.386 50-50s-22.386-50-50-50-50 22.386-50 50 22.386 50 50 50zm0 50c55.228 0 100-44.772 100-100S155.228 0 100 0 0 44.772 0 100s44.772 100 100 100z"
                  clipRule="evenodd"
                ></path>
              </mask>
              <g mask="url(#cs_mask_1_ellipse-12)">
                <path fill="#fff" d="M200 0H0v200h200V0z"></path>
                <path
                  fill="hsl(var(--primary))"
                  fillOpacity="0.33"
                  d="M200 0H0v200h200V0z"
                ></path>
                <motion.g
                  filter="url(#filter0_f_844_2811)"
                  animate={{
                    opacity: [0.6, 1, 0.6],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                >
                  <path
                    fill="hsl(var(--primary))"
                    d="M110 32H18v68h92V32z"
                  ></path>
                  <path
                    fill="hsl(var(--primary))"
                    d="M188-24H15v98h173v-98z"
                  ></path>
                  <path
                    fill="hsl(var(--primary) / 0.8)"
                    d="M175 70H5v156h170V70z"
                  ></path>
                  <path
                    fill="hsl(var(--primary) / 0.6)"
                    d="M230 51H100v103h130V51z"
                  ></path>
                </motion.g>
              </g>
            </g>
            <defs>
              <filter
                id="filter0_f_844_2811"
                width="385"
                height="410"
                x="-75"
                y="-104"
                colorInterpolationFilters="sRGB"
                filterUnits="userSpaceOnUse"
              >
                <feFlood floodOpacity="0" result="BackgroundImageFix"></feFlood>
                <feBlend
                  in="SourceGraphic"
                  in2="BackgroundImageFix"
                  result="shape"
                ></feBlend>
                <feGaussianBlur
                  result="effect1_foregroundBlur_844_2811"
                  stdDeviation="40"
                ></feGaussianBlur>
              </filter>
              <clipPath id="cs_clip_1_ellipse-12">
                <path fill="#fff" d="M0 0H200V200H0z"></path>
              </clipPath>
            </defs>
            <g
              style={{ mixBlendMode: "overlay" }}
              mask="url(#cs_mask_1_ellipse-12)"
            >
              <path
                fill="gray"
                stroke="transparent"
                d="M200 0H0v200h200V0z"
                filter="url(#cs_noise_1_ellipse-12)"
              ></path>
            </g>
            <defs>
              <filter
                id="cs_noise_1_ellipse-12"
                width="100%"
                height="100%"
                x="0%"
                y="0%"
                filterUnits="objectBoundingBox"
              >
                <feTurbulence
                  baseFrequency="0.6"
                  numOctaves="5"
                  result="out1"
                  seed="4"
                ></feTurbulence>
                <feComposite
                  in="out1"
                  in2="SourceGraphic"
                  operator="in"
                  result="out2"
                ></feComposite>
                <feBlend
                  in="SourceGraphic"
                  in2="out2"
                  mode="overlay"
                  result="out3"
                ></feBlend>
              </filter>
            </defs>
          </motion.svg>
        </motion.div>

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
          className="w-full grid grid-cols-3 gap-4 mb-4"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.6 }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1.8 }}
          >
            <CommandButton
              icon={<BookOpen className="w-5 h-5" />}
              label="Learn"
              isActive={activeCommandCategory === "learn"}
              onClick={() =>
                setActiveCommandCategory(
                  activeCommandCategory === "learn" ? null : "learn",
                )
              }
            />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 2.0 }}
          >
            <CommandButton
              icon={<Code className="w-5 h-5" />}
              label="Code"
              isActive={activeCommandCategory === "code"}
              onClick={() =>
                setActiveCommandCategory(
                  activeCommandCategory === "code" ? null : "code",
                )
              }
            />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 2.2 }}
          >
            <CommandButton
              icon={<PenTool className="w-5 h-5" />}
              label="Write"
              isActive={activeCommandCategory === "write"}
              onClick={() =>
                setActiveCommandCategory(
                  activeCommandCategory === "write" ? null : "write",
                )
              }
            />
          </motion.div>
        </motion.div>

        {/* Command suggestions */}
        <AnimatePresence>
          {activeCommandCategory && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="w-full overflow-hidden"
            >
              <div className="bg-background rounded-xl border border-border shadow-sm overflow-hidden">
                <div className="p-3 border-b border-border">
                  <h3 className="text-sm font-medium text-foreground">
                    {activeCommandCategory === "learn"
                      ? "Learning suggestions"
                      : activeCommandCategory === "code"
                        ? "Coding suggestions"
                        : "Writing suggestions"}
                  </h3>
                </div>
                <ul className="divide-y divide-border">
                  {commandSuggestions[
                    activeCommandCategory as keyof typeof commandSuggestions
                  ].map((suggestion, index) => (
                    <motion.li
                      key={index}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.03 }}
                      onClick={() => handleCommandSelect(suggestion)}
                      className="p-3 hover:bg-muted cursor-pointer transition-colors duration-75"
                    >
                      <div className="flex items-center gap-3">
                        {activeCommandCategory === "learn" ? (
                          <BookOpen className="w-4 h-4 text-primary" />
                        ) : activeCommandCategory === "code" ? (
                          <Code className="w-4 h-4 text-primary" />
                        ) : (
                          <PenTool className="w-4 h-4 text-primary" />
                        )}
                        <span className="text-sm text-foreground">
                          {suggestion}
                        </span>
                      </div>
                    </motion.li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}
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
            onClick={() => sendMessage({ text: suggestion })}
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
