import { Chat } from "./core/chat";

export interface AdaptiveClientConfig {
  apiKey?: string;
  baseUrl?: string;
}

export class Adaptive {
  public chat: Chat;

  constructor(config: AdaptiveClientConfig = {}) {
    const apiKey =
      config.apiKey ||
      (typeof process !== "undefined" && process.env?.ADAPTIVE_API_KEY) ||
      (typeof import.meta !== "undefined" &&
        import.meta.env?.VITE_ADAPTIVE_API_KEY);
    const baseUrl =
      config.baseUrl ||
      (typeof process !== "undefined" && process.env?.ADAPTIVE_BASE_URL) ||
      (typeof import.meta !== "undefined" &&
        import.meta.env?.VITE_ADAPTIVE_BASE_URL) ||
      "https://backend-go.salmonwave-ec8d1f2a.eastus.azurecontainerapps.io";

    if (!apiKey) {
      throw new Error(
        "Missing API key. Set it via config or ADAPTIVE_API_KEY env var.",
      );
    }

    if (!/^https?:\/\//.test(baseUrl)) {
      throw new Error(`Invalid baseUrl: ${baseUrl}`);
    }

    this.chat = new Chat(apiKey, baseUrl);
  }
}
