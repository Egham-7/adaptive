"use client"

import { cn } from "@/lib/shared/utils"
import React, { useEffect, useState } from "react"
import { codeToHtml } from "shiki"
import { useTheme } from "next-themes"

export type CodeBlockProps = {
  children?: React.ReactNode
  className?: string
} & React.HTMLProps<HTMLDivElement>

function CodeBlock({ children, className, ...props }: CodeBlockProps) {
  return (
    <div
      className={cn(
        "not-prose flex w-full flex-col overflow-clip border",
        "border-border bg-card text-card-foreground rounded-xl",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export type CodeBlockCodeProps = {
  code: string
  language?: string
  theme?: string
  className?: string
} & React.HTMLProps<HTMLDivElement>

function CodeBlockCode({
  code,
  language = "tsx",
  theme,
  className,
  ...props
}: CodeBlockCodeProps) {
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null)
  const { resolvedTheme } = useTheme()

  useEffect(() => {
    async function highlight() {
      if (!code) {
        setHighlightedHtml("<pre><code></code></pre>")
        return
      }

      // Skip highlighting until theme is resolved to prevent flash
      if (theme === undefined && !resolvedTheme) {
        return
      }

      // Auto-detect theme based on current theme if not explicitly provided
      const selectedTheme = theme || (resolvedTheme === 'dark' ? 'github-dark' : 'github-light')
      
      const html = await codeToHtml(code, { 
        lang: language, 
        theme: selectedTheme,
        transformers: []
      })
      setHighlightedHtml(html)
    }
    highlight()
  }, [code, language, theme, resolvedTheme])

  const classNames = cn(
    "w-full overflow-x-auto text-[13px] [&>pre]:px-4 [&>pre]:py-4",
    className
  )

  // SSR fallback: render plain code if not hydrated yet
  return highlightedHtml ? (
    <div
      className={classNames}
      dangerouslySetInnerHTML={{ __html: highlightedHtml }}
      role="region"
      aria-label={`Code block in ${language}`}
      {...props}
    />
  ) : (
    <div className={classNames} role="region" aria-label={`Code block in ${language}`} {...props}>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  )
}

export type CodeBlockGroupProps = React.HTMLAttributes<HTMLDivElement>

function CodeBlockGroup({
  children,
  className,
  ...props
}: CodeBlockGroupProps) {
  return (
    <div
      className={cn("flex items-center justify-between", className)}
      {...props}
    >
      {children}
    </div>
  )
}

export { CodeBlockGroup, CodeBlockCode, CodeBlock }
