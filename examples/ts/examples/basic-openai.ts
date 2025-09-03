import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "sk-vusJ4V3qwk7q4jk_FCXpPjINnc34vNmWG1lGH1m85Q7KEEoL",
  baseURL: "http://localhost:3000/api/v1",
});

async function main() {
  try {
    const response = await client.chat.completions.create({
      model: "", // Leave empty for intelligent routing
      messages: [{ role: "user", content: "Hello!" }],
    });

    if (!response.choices?.[0]?.message?.content) {
      console.log("No response content received");
      return;
    }

    console.log(response.choices[0].message.content);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
