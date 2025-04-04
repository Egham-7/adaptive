import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import { cn } from "@/lib/utils";
import { Copy, Check } from "lucide-react";

interface MarkdownProps {
  content: string;
  className?: string;
}

export default function Markdown({ content, className }: MarkdownProps) {
  return (
    <div
      className={cn(
        "prose prose-sm dark:prose-invert w-full max-w-none overflow-hidden break-words",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        components={{
          a: (props) => (
            <a
              {...props}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:underline break-words"
            />
          ),
          code(props) {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const { children, className, ref, ...rest } = props;
            const match = /language-(\w+)/.exec(className || "");

            // Handle inline code (not in a code block)
            if (!match) {
              return (
                <code
                  {...rest}
                  className="bg-muted-foreground/20 px-1 py-0.5 rounded text-sm break-words whitespace-pre-wrap"
                >
                  {children}
                </code>
              );
            }

            // Handle code blocks
            const language = match[1];
            const codeString = String(children).replace(/\n$/, "");

            return (
              <CodeBlock language={language} value={codeString}>
                <SyntaxHighlighter
                  {...rest}
                  PreTag="div"
                  language={language}
                  style={oneDark}
                  customStyle={{
                    margin: 0,
                    borderRadius: 0,
                    background: "transparent",
                    overflowX: "auto",
                    wordBreak: "normal",
                    wordWrap: "normal",
                  }}
                  wrapLines={true}
                  wrapLongLines={true}
                  showLineNumbers={false}
                >
                  {codeString}
                </SyntaxHighlighter>
              </CodeBlock>
            );
          },
          pre: ({ children, ...props }) => (
            <pre
              className="overflow-auto w-full rounded-md bg-transparent p-0 m-0"
              {...props}
            >
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto w-full">
              <table className="w-full border-collapse table-auto">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="px-4 py-2 text-left border border-slate-300 dark:border-slate-700">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2 border border-slate-300 dark:border-slate-700 break-words">
              {children}
            </td>
          ),
          p: ({ children }) => (
            <p className="break-words whitespace-pre-wrap my-4">{children}</p>
          ),
          ul: ({ children }) => (
            <ul className="pl-6 list-disc space-y-2 my-4">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="pl-6 list-decimal space-y-2 my-4">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="break-words mb-1">{children}</li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="pl-4 border-l-4 border-muted-foreground/30 italic text-muted-foreground my-4">
              {children}
            </blockquote>
          ),
          h1: ({ children }) => (
            <h1 className="text-2xl font-bold mt-8 mb-4 break-words">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl font-bold mt-7 mb-3 break-words">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg font-bold mt-6 mb-2 break-words">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-base font-bold mt-5 mb-2 break-words">
              {children}
            </h4>
          ),
          h5: ({ children }) => (
            <h5 className="text-sm font-bold mt-4 mb-1 break-words">
              {children}
            </h5>
          ),
          h6: ({ children }) => (
            <h6 className="text-xs font-bold mt-4 mb-1 break-words">
              {children}
            </h6>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

interface CodeBlockProps {
  language: string;
  value: string;
  children: React.ReactNode;
}

function CodeBlock({ language, value, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative my-4 rounded-md group bg-muted-foreground/10 w-full">
      {language && (
        <div className="px-3 py-1 font-mono text-xs bg-muted-foreground/20 text-muted-foreground rounded-t-md flex justify-between items-center">
          <span>{language}</span>
          <button
            onClick={handleCopy}
            className="p-1 rounded-md text-muted-foreground hover:bg-muted transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
            aria-label="Copy code"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      )}
      <div className="p-4 overflow-x-auto w-full scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
        {children}
      </div>
      {!language && (
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-1.5 rounded-md bg-background/80 text-muted-foreground hover:bg-muted transition-colors focus:outline-none focus:ring-2 focus:ring-primary opacity-0 group-hover:opacity-100"
          aria-label="Copy code"
        >
          {copied ? (
            <Check className="w-4 h-4" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
        </button>
      )}
    </div>
  );
}
