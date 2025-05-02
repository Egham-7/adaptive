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
    <div className={cn("prose prose-sm dark:prose-invert w-full max-w-none", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        components={{
          a: (props) => (
            <a 
              {...props} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-blue-500 hover:underline"
            />
          ),
          code(props) {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const { children, className, ref, ...rest } = props;
            const match = /language-(\w+)/.exec(className || '');
            
            if (!match) {
              return (
                <code 
                  {...rest} 
                  className="bg-muted-foreground/20 px-1 py-0.5 rounded text-sm"
                >
                  {children}
                </code>
              );
            }

            const language = match[1];
            const codeString = String(children).replace(/\n$/, '');
            
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
                    background: 'transparent' 
                  }}
                >
                  {codeString}
                </SyntaxHighlighter>
              </CodeBlock>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto">
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
            <td className="px-4 py-2 border border-slate-300 dark:border-slate-700">
              {children}
            </td>
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
    <div className="relative my-4 rounded-md group bg-muted-foreground/10">
      {language && (
        <div className="px-3 py-1 font-mono text-xs bg-muted-foreground/20 text-muted-foreground rounded-t-md">
          {language}
        </div>
      )}
      <div className="p-4 overflow-auto">
        {children}
      </div>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-1.5 rounded-md bg-background/80 text-muted-foreground hover:bg-muted transition-colors focus:outline-hidden focus:ring-2 focus:ring-primary opacity-0 group-hover:opacity-100"
        aria-label="Copy code"
      >
        {copied ? (
          <Check className="w-4 h-4" />
        ) : (
          <Copy className="w-4 h-4" />
        )}
      </button>
    </div>
  );
}