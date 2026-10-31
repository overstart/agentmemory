import type { EmbeddingProvider } from "../../types.js";
import { getEnvVar } from "../../config.js";
import { fetchWithTimeout } from "../_fetch.js";
import {
  DEFAULT_AZURE_API_VERSION,
  buildAuthHeaders,
  buildEmbeddingUrl,
  detectAzure,
  normalizeBaseUrl,
} from "../_openai-shared.js";
import { resolveDimensions } from "./_dimensions.js";

const DEFAULT_MODEL = "text-embedding-3-small";

// Optional process-wide throttle for strict per-account embedding rate limits
// (e.g. Volcengine coding plan rejects startup backfill bursts with 429 while
// sequential calls at 60 rpm succeed). Serializes every batch and enforces a
// minimum gap between request starts. shortcut: only active when the env var is set
let embGate: Promise<void> = Promise.resolve();
let lastEmbStart = 0;
async function embeddingGate(): Promise<void> {
  const minMs =
    parseInt(getEnvVar("OPENAI_EMBEDDING_MIN_INTERVAL_MS") || "0", 10) || 0;
  // Disabled by default: bypass the chain entirely so ordinary users pay no
  // serialization cost (reads the env per call, so the knob is live without
  // rebuilding the provider).
  if (minMs <= 0) return;
  const run = embGate.then(async () => {
    const wait = lastEmbStart + minMs - Date.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    lastEmbStart = Date.now();
  });
  embGate = run.catch(() => {});
  return run;
}

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai";
  readonly dimensions: number;
  private apiKey: string;
  private baseUrl: string;
  private model: string;
  private isAzure: boolean;
  private azureApiVersion: string;

  constructor(apiKey?: string) {
    // Separate API key path: caller-passed wins, then OPENAI_EMBEDDING_API_KEY,
    // then fall back to OPENAI_API_KEY. Allows e.g. a placeholder key for
    // local endpoints that ignore Authorization (most do).
    this.apiKey =
      apiKey?.trim() ||
      getEnvVar("OPENAI_EMBEDDING_API_KEY")?.trim() ||
      getEnvVar("OPENAI_API_KEY")?.trim() ||
      "";
    if (!this.apiKey) {
      throw new Error(
        "API key is required (via constructor, OPENAI_EMBEDDING_API_KEY, or OPENAI_API_KEY)",
      );
    }
    // Embedding-specific base URL override; falls back to OPENAI_BASE_URL,
    // then normalizeBaseUrl's default. The chat-LLM path (src/providers/openai.ts)
    // still reads only OPENAI_BASE_URL, so setting OPENAI_EMBEDDING_BASE_URL
    // alone moves embeddings to the new endpoint without affecting chat.
    this.baseUrl = normalizeBaseUrl(
      getEnvVar("OPENAI_EMBEDDING_BASE_URL") || getEnvVar("OPENAI_BASE_URL"),
    );
    this.model = getEnvVar("OPENAI_EMBEDDING_MODEL") || DEFAULT_MODEL;
    this.dimensions = resolveDimensions(
      this.model,
      getEnvVar("OPENAI_EMBEDDING_DIMENSIONS"),
      "OPENAI_EMBEDDING_DIMENSIONS",
    );
    this.isAzure = detectAzure(this.baseUrl);
    this.azureApiVersion =
      getEnvVar("OPENAI_API_VERSION") || DEFAULT_AZURE_API_VERSION;
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    await embeddingGate();
    const url = buildEmbeddingUrl(
      this.baseUrl,
      this.isAzure,
      this.azureApiVersion,
    );
    const response = await fetchWithTimeout(url, {
      method: "POST",
      headers: buildAuthHeaders(this.apiKey, this.isAzure),
      body: JSON.stringify({
        model: this.model,
        input: texts,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`OpenAI embedding failed (${response.status}): ${err}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    return data.data.map((d) => new Float32Array(d.embedding));
  }
}
