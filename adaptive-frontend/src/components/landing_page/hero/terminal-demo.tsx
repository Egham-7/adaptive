import PythonTerminalDemo from "./terminal-demo/python";

export default function LanguageTerminalDemo({
  language,
}: {
  language: string;
}) {
  switch (language.toLowerCase()) {
    case "python":
      return <PythonTerminalDemo />;
    default:
      throw new Error(`Language ${language} not supported`);
  }
}
