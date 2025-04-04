import {
  AnimatedSpan,
  TypingAnimation,
  Terminal,
} from "@/components/magicui/terminal";

export default function PythonTerminalDemo() {
  return (
    <Terminal className="min-w-full max-h-full">
      <TypingAnimation>{"> pip install adaptive"}</TypingAnimation>
      <AnimatedSpan delay={1500} className="text-green-500">
        <span>✔ Collecting adaptive</span>
      </AnimatedSpan>
      <AnimatedSpan delay={2500} className="text-green-500">
        <span>✔ Successfully installed adaptive-0.14.2</span>
      </AnimatedSpan>

      <TypingAnimation delay={3500}>{"> python"}</TypingAnimation>
      <AnimatedSpan delay={4000} className="text-blue-500">
        <span>Python 3.11.4</span>
      </AnimatedSpan>

      <TypingAnimation delay={4500}>
        {">>> from adaptive import Adaptive"}
      </TypingAnimation>
      <TypingAnimation delay={5000}>
        {">>> adaptive = Adaptive()"}
      </TypingAnimation>
      <TypingAnimation delay={5500}>
        {">>> async def generate_text(prompt):"}
      </TypingAnimation>
      <TypingAnimation delay={6000}>
        {"...     response = await adaptive.chat.completions.create("}
      </TypingAnimation>
      <TypingAnimation delay={6500}>
        {'...         messages=[{"role": "user", "content": prompt}],'}
      </TypingAnimation>
      <TypingAnimation delay={7000}>{"...     )"}</TypingAnimation>
      <TypingAnimation delay={7500}>{"...     "}</TypingAnimation>
      <TypingAnimation delay={8000}>
        {"...     return response.choices[0].message.content"}
      </TypingAnimation>

      <AnimatedSpan delay={8500} className="text-green-500">
        <span>✔ Function defined successfully</span>
      </AnimatedSpan>

      <TypingAnimation delay={9000} className="text-muted-foreground">
        {"Success! Adaptive is ready for text generation."}
      </TypingAnimation>
    </Terminal>
  );
}
