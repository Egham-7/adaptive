import React from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { CodeBlock, CodeBlockCode } from "@/components/ui/code-block";
import { CopyButton } from "@/components/ui/copy-button";
import { cn } from "@/lib/shared/utils";
import { useTheme } from "next-themes";

interface MarkdownRendererProps {
  children: string;
}

export function MarkdownRenderer({ children }: MarkdownRendererProps) {
  return (
    <div className="space-y-3">
      <Markdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children}
      </Markdown>
    </div>
  );
}

interface CustomCodeBlockProps extends React.HTMLAttributes<HTMLPreElement> {
  children: React.ReactNode;
  className?: string;
  language: string;
}

const CustomCodeBlock = ({
  children,
  className,
  language,
}: CustomCodeBlockProps) => {
  const code =
    typeof children === "string"
      ? children
      : childrenTakeAllStringContents(children);

  const { resolvedTheme } = useTheme();

  const codeTheme = resolvedTheme === "dark" ? "github-dark" : "github-light";

  return (
    <div className="group/code relative mb-4">
      <CodeBlock className={className}>
        <CodeBlockCode
          code={code}
          language={language}
          theme={codeTheme}
          className="relative"
        />
        <div className="invisible absolute top-2 right-2 flex space-x-1 rounded-lg bg-background/80 backdrop-blur-sm p-1 opacity-0 transition-all duration-200 group-hover/code:visible group-hover/code:opacity-100">
          <CopyButton content={code} copyMessage="Copied code to clipboard" />
        </div>
      </CodeBlock>
    </div>
  );
};

function childrenTakeAllStringContents(element: React.ReactNode): string {
  if (typeof element === "string") {
    return element;
  }

  if (React.isValidElement(element)) {
    const props = element.props as Record<string, unknown>;
    if (props && typeof props === "object" && "children" in props) {
      const children = props.children as React.ReactNode;

      if (Array.isArray(children)) {
        return children
          .map((child) => childrenTakeAllStringContents(child))
          .join("");
      }
      return childrenTakeAllStringContents(children);
    }
  }

  return "";
}

const COMPONENTS = {
  h1: withClass("h1", "text-2xl font-semibold"),
  h2: withClass("h2", "font-semibold text-xl"),
  h3: withClass("h3", "font-semibold text-lg"),
  h4: withClass("h4", "font-semibold text-base"),
  h5: withClass("h5", "font-medium"),
  strong: withClass("strong", "font-semibold"),
  a: withClass("a", "text-primary underline underline-offset-2"),
  blockquote: withClass("blockquote", "border-l-2 border-primary pl-4"),
  code: ({
    children,
    className,
    node,
    ...rest
  }: React.PropsWithChildren<{ className?: string; node?: unknown }>) => {
    const match = /language-(\w+)/.exec(className || "");
    return match ? (
      <CustomCodeBlock
        className={className}
        language={match[1] ?? ""}
        {...rest}
      >
        {children}
      </CustomCodeBlock>
    ) : (
      <code
        className={cn(
          "font-mono [:not(pre)>&]:rounded-md [:not(pre)>&]:bg-background/50 [:not(pre)>&]:px-1 [:not(pre)>&]:py-0.5",
        )}
        {...rest}
      >
        {children}
      </code>
    );
  },
  pre: ({ children }: React.PropsWithChildren) => children,
  ol: withClass("ol", "list-decimal space-y-2 pl-6"),
  ul: withClass("ul", "list-disc space-y-2 pl-6"),
  li: withClass("li", "my-1.5"),
  table: withClass(
    "table",
    "w-full border-collapse overflow-y-auto rounded-md border border-foreground/20",
  ),
  th: withClass(
    "th",
    "border border-foreground/20 px-4 py-2 text-left font-bold [&[align=center]]:text-center [&[align=right]]:text-right",
  ),
  td: withClass(
    "td",
    "border border-foreground/20 px-4 py-2 text-left [&[align=center]]:text-center [&[align=right]]:text-right",
  ),
  tr: withClass("tr", "m-0 border-t p-0 even:bg-muted"),
  p: withClass("p", "whitespace-pre-wrap"),
  hr: withClass("hr", "border-foreground/20"),
};

function withClass(Tag: keyof React.JSX.IntrinsicElements, classes: string) {
  const Component = ({ node, ...props }: { node?: unknown }) => {
    const Element = Tag as keyof React.JSX.IntrinsicElements;
    return React.createElement(Element, { className: classes, ...props });
  };
  Component.displayName = Tag;
  return Component;
}

export default MarkdownRenderer;
