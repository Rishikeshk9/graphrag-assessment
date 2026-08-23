import type {
  ChatTurn,
  Citation,
  GraphEdge,
  GraphNode,
  GraphTriple,
  ParentContext,
  Subgraph,
} from "@/types";

export const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8080/api/v1";

export type Fetcher = typeof fetch;

export type SseFrame = { event: string; data: unknown };

export function parseSseFrame(frame: string): SseFrame | null {
  const event = frame.match(/^event: (.+)$/m)?.[1];
  const raw = frame.match(/^data: (.*)$/m)?.[1];
  if (!event || raw === undefined) return null;
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return null;
  }
}

export async function* readSseFrames(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const parsed = parseSseFrame(frame);
      if (parsed) yield parsed;
    }
    if (done) break;
  }
}

export type ChatPayload = {
  query: string;
  history: ChatTurn[];
  graphHops?: number;
  llmProvider?: ModelProvider;
};

export type ModelProvider = "local" | "openrouter";

export type ModelProviders = {
  default_provider: ModelProvider;
  openrouter_configured: boolean;
  embedding_provider: "local";
};

export type ChatHandlers = {
  onSources?: (sources: Citation[]) => void;
  onParents?: (parents: ParentContext[]) => void;
  onGraph?: (triples: GraphTriple[]) => void;
  onToken?: (token: string) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
};

export async function streamChat(
  payload: ChatPayload,
  handlers: ChatHandlers,
  options: { fetchImpl?: Fetcher; signal?: AbortSignal } = {},
): Promise<void> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      query: payload.query,
      history: payload.history,
      graph_hops: payload.graphHops ?? 2,
      llm_provider: payload.llmProvider ?? "local",
    }),
    signal: options.signal,
  });
  if (!response.ok || !response.body)
    throw new Error(`Chat request failed (${response.status})`);

  for await (const frame of readSseFrames(response.body)) {
    switch (frame.event) {
      case "sources":
        handlers.onSources?.(frame.data as Citation[]);
        break;
      case "parents":
        handlers.onParents?.(frame.data as ParentContext[]);
        break;
      case "graph":
        handlers.onGraph?.(frame.data as GraphTriple[]);
        break;
      case "token":
        handlers.onToken?.(frame.data as string);
        break;
      case "error":
        handlers.onError?.(frame.data as string);
        break;
      case "done":
        handlers.onDone?.();
        break;
    }
  }
}

export type IngestJob = {
  job_id: string;
  status: "accepted" | "processing" | "completed" | "failed" | "cancelled";
  phase: string;
  child_chunks_indexed: number;
  graph_relationships_indexed: number;
  graph_children_processed: number;
  graph_children_total: number;
  warnings?: string[];
  detail?: string;
};

export type ClearKnowledgeBaseResult = {
  vectors_removed: number;
  relationships_removed: number;
  entities_removed: number;
};

export type KnowledgeBaseUsage = {
  qdrant_parent_vectors: number;
  qdrant_child_vectors: number;
  neo4j_entities: number;
  neo4j_relationships: number;
};

export type KnowledgeBaseDocument = {
  source_id: string;
  providers: ModelProvider[];
  parent_vectors: number;
  child_vectors: number;
  file_sha256?: string | null;
};

export type IngestOptions = {
  fetchImpl?: Fetcher;
  /** Optional test/automation safety cap. Interactive indexing has no deadline. */
  maxAttempts?: number;
  intervalMs?: number;
  wait?: (ms: number) => Promise<void>;
  llmProvider?: ModelProvider;
  signal?: AbortSignal;
  onStarted?: (job: IngestJob) => void;
};

const delay = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

async function readJsonResponse(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    if (response.status === 413) {
      return { detail: "The PDF is larger than the server upload limit." };
    }
    return { detail: `The server returned an unexpected response (HTTP ${response.status}).` };
  }
}

export async function ingestPdf(
  file: File,
  onProgress: (job: IngestJob) => void,
  options: IngestOptions = {},
): Promise<IngestJob> {
  const request = options.fetchImpl ?? fetch;
  const maxAttempts = options.maxAttempts;
  const intervalMs = options.intervalMs ?? 1_200;
  const wait = options.wait ?? delay;

  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({
    llm_provider: options.llmProvider ?? "local",
  });
  const created = await request(`${API_BASE}/ingest/file?${params}`, {
    method: "POST",
    body: form,
    signal: options.signal,
  });
  const job = (await readJsonResponse(created)) as IngestJob;
  if (!created.ok) throw new Error(job.detail ?? "Upload failed");
  options.onStarted?.(job);

  let attempt = 0;
  // A document can take longer than any arbitrary browser-side deadline,
  // especially while graph extraction is using a local model. Keep polling
  // until the API reports completion, cancellation, or a real failure. The
  // caller may still provide maxAttempts for a controlled test or automation.
  while (maxAttempts === undefined || attempt < maxAttempts) {
    attempt += 1;
    const response = await request(`${API_BASE}/ingest/${job.job_id}`, { signal: options.signal });
    const result = (await readJsonResponse(response)) as IngestJob;
    if (!response.ok)
      throw new Error(result.detail ?? "Could not read indexing progress");
    if (result.status === "completed") return result;
    if (result.status === "cancelled")
      throw new Error("Indexing was cancelled");
    if (result.status === "failed")
      throw new Error(result.warnings?.[0] ?? "Indexing failed");
    onProgress(result);
    await wait(intervalMs);
  }
  throw new Error(
    `Indexing is still running after ${Math.round((maxAttempts ?? 0) * intervalMs / 1000)}s. Check the API logs.`,
  );
}

export async function cancelIngestion(
  jobId: string,
  options: { fetchImpl?: Fetcher } = {},
): Promise<IngestJob> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/ingest/${jobId}/cancel`, { method: "POST" });
  const result = (await readJsonResponse(response)) as IngestJob;
  if (!response.ok) throw new Error(result.detail ?? "Could not cancel indexing");
  return result;
}

export async function clearKnowledgeBase(
  options: { fetchImpl?: Fetcher } = {},
): Promise<ClearKnowledgeBaseResult> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/knowledge-base`, { method: "DELETE" });
  const result = (await readJsonResponse(response)) as ClearKnowledgeBaseResult & { detail?: string };
  if (!response.ok) throw new Error(result.detail ?? "Could not clear the knowledge base");
  return result;
}

export async function fetchKnowledgeBaseUsage(
  options: { fetchImpl?: Fetcher } = {},
): Promise<KnowledgeBaseUsage> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/knowledge-base/usage`);
  const result = (await readJsonResponse(response)) as KnowledgeBaseUsage & { detail?: string };
  if (!response.ok) throw new Error(result.detail ?? "Could not load knowledge-base usage");
  return result;
}

export async function fetchKnowledgeBaseDocuments(
  options: { fetchImpl?: Fetcher } = {},
): Promise<KnowledgeBaseDocument[]> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/knowledge-base/documents`);
  const result = (await readJsonResponse(response)) as { documents?: KnowledgeBaseDocument[]; detail?: string };
  if (!response.ok) throw new Error(result.detail ?? "Could not load indexed documents");
  return result.documents ?? [];
}

export async function deleteKnowledgeBaseDocument(
  sourceId: string,
  options: { fetchImpl?: Fetcher } = {},
): Promise<{ source_id: string; vectors_removed: number; relationships_removed: number }> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/documents/${encodeURIComponent(sourceId)}`, { method: "DELETE" });
  const result = (await readJsonResponse(response)) as { detail?: string; source_id: string; vectors_removed: number; relationships_removed: number };
  if (!response.ok) throw new Error(result.detail ?? "Could not remove document");
  return result;
}

export async function fetchModelProviders(
  options: { fetchImpl?: Fetcher } = {},
): Promise<ModelProviders> {
  const request = options.fetchImpl ?? fetch;
  const response = await request(`${API_BASE}/health/model-providers`);
  if (!response.ok) throw new Error(`Could not load model providers (${response.status})`);
  return (await response.json()) as ModelProviders;
}

export async function fetchSubgraph(
  query: string,
  options: { fetchImpl?: Fetcher; graphHops?: number; signal?: AbortSignal } = {},
): Promise<Subgraph> {
  const request = options.fetchImpl ?? fetch;
  const params = new URLSearchParams({
    query,
    graph_hops: String(options.graphHops ?? 2),
  });
  const response = await request(`${API_BASE}/graph/subgraph?${params}`, {
    signal: options.signal,
  });
  if (!response.ok) throw new Error(`Subgraph request failed (${response.status})`);
  return (await response.json()) as Subgraph;
}

/** Fallback view built from the triples already streamed with the answer. */
export function subgraphFromTriples(triples: GraphTriple[]): Subgraph {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = triples.map((triple, index) => {
    nodes.set(triple.subject, { id: triple.subject, label: triple.subject });
    nodes.set(triple.object, { id: triple.object, label: triple.object });
    return {
      id: `${index}:${triple.source_child_chunk_id}`,
      source: triple.subject,
      target: triple.object,
      label: triple.predicate,
      evidence: triple.evidence,
    };
  });
  return { nodes: [...nodes.values()], edges };
}
