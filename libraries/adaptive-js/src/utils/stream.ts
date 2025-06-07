import { ZodSchema } from "zod";

export interface StreamIteratorOptions<T> {
  schema?: ZodSchema<T>;
  doneSignal?: string;
}

/**
 * Generic async stream iterator for SSE/streaming JSON chunks.
 */
export async function* createStreamIterator<T>(
  reader: ReadableStreamDefaultReader<Uint8Array> | undefined,
  options?: StreamIteratorOptions<T>,
): AsyncGenerator<T> {
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;

      const raw = line.slice(5).trim();
      if (raw === (options?.doneSignal ?? "[DONE]")) return;

      try {
        const json = JSON.parse(raw);

        if (options?.schema) {
          const parsed = options.schema.parse(json);
          yield parsed;
        } else {
          yield json as T;
        }
      } catch (err) {
        console.warn("Stream parse error:", err);
        continue;
      }
    }
  }
}
