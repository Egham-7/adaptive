from openai import OpenAI

client = OpenAI(
    api_key="sk-HWc1dstDuA60xkqwbHLpuKM01_OmZqqnurJbH4KVLc-Xo0Ns",
    base_url="http://localhost:3000/api/v1",
)

completion = client.chat.completions.create(
    model="",  # Leave empty for intelligent routing
    messages=[{"role": "user", "content": "Explain quantum computing simply"}],
)

print(completion.choices[0].message.content)
