import { FileIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { useTheme } from "@/context/theme-provider";
import { diffLines, Change } from "diff";

interface CodeComparisonProps {
  beforeCode: string;
  afterCode: string;
  language: string;
  filename: string;
  lightTheme: string;
  darkTheme: string;
}

export function CodeComparison({
  beforeCode,
  afterCode,
  language,
  filename,
  lightTheme,
  darkTheme,
}: CodeComparisonProps) {
  const { theme } = useTheme();
  const [diffResult, setDiffResult] = useState<Change[]>([]);
  const [highlightedBeforeParts, setHighlightedBeforeParts] = useState<
    Record<string, string>
  >({});
  const [highlightedAfterParts, setHighlightedAfterParts] = useState<
    Record<string, string>
  >({});

  useEffect(() => {
    // Calculate diff
    const diff = diffLines(beforeCode, afterCode);
    setDiffResult(diff);

    const selectedTheme = theme === "dark" ? darkTheme : lightTheme;

    async function highlightCode() {
      try {
        const { codeToHtml } = await import("shiki");

        // Highlight each part separately
        const beforeParts: Record<string, string> = {};
        const afterParts: Record<string, string> = {};

        // Process before parts (removed and unchanged)
        for (let i = 0; i < diff.length; i++) {
          const part = diff[i];

          if (part.removed || (!part.added && !part.removed)) {
            const lines = part.value.split("\n").filter(Boolean);
            for (let j = 0; j < lines.length; j++) {
              const key = `${i}-${j}`;
              const html = await codeToHtml(lines[j], {
                lang: language,
                theme: selectedTheme,
              });
              beforeParts[key] = html;
            }
          }

          if (part.added || (!part.added && !part.removed)) {
            const lines = part.value.split("\n").filter(Boolean);
            for (let j = 0; j < lines.length; j++) {
              const key = `${i}-${j}`;
              const html = await codeToHtml(lines[j], {
                lang: language,
                theme: selectedTheme,
              });
              afterParts[key] = html;
            }
          }
        }

        setHighlightedBeforeParts(beforeParts);
        setHighlightedAfterParts(afterParts);
      } catch (error) {
        console.error("Error highlighting code:", error);
      }
    }

    highlightCode();
  }, [theme, beforeCode, afterCode, language, lightTheme, darkTheme]);

  const renderDiffLine = (
    code: string,
    isRemoved: boolean,
    isAdded: boolean,
    highlightedHtml: string | undefined,
  ) => {
    const bgClass = isRemoved
      ? "bg-red-500/10"
      : isAdded
        ? "bg-green-500/10"
        : "";

    const indicatorClass = isRemoved
      ? "text-red-500"
      : isAdded
        ? "text-green-500"
        : "text-gray-400";

    const indicator = isRemoved ? "-" : isAdded ? "+" : " ";

    return (
      <div className={`flex ${bgClass} whitespace-pre`}>
        <span
          className={`select-none w-5 inline-block ${indicatorClass} flex-shrink-0`}
        >
          {indicator}
        </span>
        {highlightedHtml ? (
          <div
            className="font-mono text-xs [&>pre]:!bg-transparent [&>pre]:!p-0 [&>pre]:!m-0"
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        ) : (
          <span className="font-mono text-xs text-foreground">{code}</span>
        )}
      </div>
    );
  };

  const renderDiff = (side: "before" | "after") => {
    const highlightedParts =
      side === "before" ? highlightedBeforeParts : highlightedAfterParts;

    return (
      <div className="h-full bg-background">
        <div className="h-full overflow-auto p-4">
          <div className="min-w-max">
            {diffResult.map((part, index) => {
              // For "before" side, show removed and unchanged parts
              if (
                side === "before" &&
                (part.removed || (!part.added && !part.removed))
              ) {
                return (
                  <div key={index}>
                    {part.value
                      .split("\n")
                      .filter(Boolean)
                      .map((line: string, lineIndex: number) => {
                        const key = `${index}-${lineIndex}`;
                        return (
                          <div key={key}>
                            {renderDiffLine(
                              line,
                              part.removed,
                              false,
                              highlightedParts[key],
                            )}
                          </div>
                        );
                      })}
                  </div>
                );
              }
              // For "after" side, show added and unchanged parts
              if (
                side === "after" &&
                (part.added || (!part.added && !part.removed))
              ) {
                return (
                  <div key={index}>
                    {part.value
                      .split("\n")
                      .filter(Boolean)
                      .map((line: string, lineIndex: number) => {
                        const key = `${index}-${lineIndex}`;
                        return (
                          <div key={key}>
                            {renderDiffLine(
                              line,
                              false,
                              part.added,
                              highlightedParts[key],
                            )}
                          </div>
                        );
                      })}
                  </div>
                );
              }
              return null;
            })}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="mx-auto w-full max-w-5xl">
      <div className="relative w-full overflow-hidden rounded-md border border-border">
        <div className="relative grid md:grid-cols-2 md:divide-x md:divide-border">
          <div>
            <div className="flex items-center bg-accent p-2 text-sm text-foreground">
              <FileIcon className="mr-2 h-4 w-4" />
              {filename}
              <span className="ml-auto">before</span>
            </div>
            {renderDiff("before")}
          </div>
          <div>
            <div className="flex items-center bg-accent p-2 text-sm text-foreground">
              <FileIcon className="mr-2 h-4 w-4" />
              {filename}
              <span className="ml-auto">after</span>
            </div>
            {renderDiff("after")}
          </div>
        </div>
        <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-md bg-accent text-xs text-foreground">
          VS
        </div>
      </div>
    </div>
  );
}
