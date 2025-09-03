import { createAdaptive } from "@adaptive-llm/adaptive-ai-provider";
import { generateText } from "ai";

const adaptive = createAdaptive({
  apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
  baseURL: "https://llmadaptive.uk/api/v1",
});

async function main() {
  try {
    // Intelligent model selection
    const { text } = await generateText({
      model: adaptive(),
      prompt: "Explain quantum computing",
    });

    if (!text) {
      console.log("No response text received");
      return;
    }

    console.log(text);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
