import { Completions } from "./completions";

export class Chat {
  public completions: Completions;

  constructor(apiKey: string, baseUrl: string) {
    this.completions = new Completions(apiKey, baseUrl);
  }
}
