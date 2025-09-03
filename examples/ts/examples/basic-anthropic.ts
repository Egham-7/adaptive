import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
  baseURL: "https://llmadaptive.uk/api",
});

async function main() {
  try {
    const response = await client.messages.create({
      model: "", // Leave empty for intelligent routing
      max_tokens: 1000,
      messages: [{ role: "user", content: "Hello!" }],
    });

    const firstContent = response.content[0];
    if (!firstContent || firstContent.type !== "text") {
      console.log("No text response content received");
      return;
    }

    console.log(firstContent.text);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
