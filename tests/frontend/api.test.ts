import { describe, expect, it, vi } from "vitest";
import {
  cancelIngestion,
  clearKnowledgeBase,
  fetchKnowledgeBaseUsage,
  fetchModelProviders,
  ingestPdf,
  parseSseFrame,
  readSseFrames,
  streamChat,
  subgraphFromTriples,
} from "@/lib/api";
import type { GraphTriple } from "@/types";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

function sseResponse(chunks: string[]) {
  return { ok: true, status: 200, body: streamOf(chunks) } as unknown as Response;
}

function jsonResponse(body: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

describe("parseSseFrame", () => {
  it("reads the event name and JSON payload", () => {
    expect(parseSseFrame('event: token\ndata: "hi"')).toEqual({
      event: "token",
      data: "hi",
    });
  });

  it("returns null for incomplete or malformed frames", () => {
    expect(parseSseFrame("data: {}")).toBeNull();
    expect(parseSseFrame("event: token\ndata: {oops")).toBeNull();
  });
});

describe("readSseFrames", () => {
  it("reassembles frames split across network chunks", async () => {
    const frames = [];
    for await (const frame of readSseFrames(
      streamOf(['event: token\ndata: "he', 'llo"\n\nevent: done\ndata: {}\n\n']),
    )) {
      frames.push(frame);
    }
    expect(frames).toEqual([
      { event: "token", data: "hello" },
      { event: "done", data: {} },
    ]);
  });
});

describe("streamChat", () => {
  it("dispatches evidence before tokens and forwards history", async () => {
    const fetchImpl = vi.fn(async (_url: URL | RequestInfo, _init?: RequestInit) =>
      sseResponse([
        'event: sources\ndata: [{"parent_chunk_id":"p1","child_chunk_id":"c1","source_id":"doc","excerpt":"x"}]\n\n',
        "event: graph\ndata: []\n\n",
        'event: token\ndata: "Answer"\n\n',
        "event: done\ndata: {}\n\n",
      ]),
    );
    const seen: string[] = [];

    await streamChat(
      {
        query: "What did it pay?",
        history: [{ role: "user", content: "Who acquired Beta?" }],
      },
      {
        onSources: () => seen.push("sources"),
        onGraph: () => seen.push("graph"),
        onToken: (token) => seen.push(token),
        onDone: () => seen.push("done"),
      },
      { fetchImpl },
    );

    expect(seen).toEqual(["sources", "graph", "Answer", "done"]);
    const body = JSON.parse(String(fetchImpl.mock.calls[0][1]?.body));
    expect(body.history).toHaveLength(1);
    expect(body.query).toBe("What did it pay?");
    expect(body.llm_provider).toBe("local");
  });

  it("throws when the API rejects the request", async () => {
    const fetchImpl = vi.fn(async () => ({ ok: false, status: 503 }) as Response);
    await expect(
      streamChat({ query: "hi", history: [] }, {}, { fetchImpl }),
    ).rejects.toThrow("503");
  });
});

describe("fetchModelProviders", () => {
  it("returns public provider capabilities", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({
        default_provider: "local",
        openrouter_configured: true,
        embedding_provider: "local",
      }),
    );

    await expect(fetchModelProviders({ fetchImpl })).resolves.toMatchObject({
      default_provider: "local",
      openrouter_configured: true,
    });
  });
});

describe("ingestPdf", () => {
  const file = new File(["%PDF"], "doc.pdf", { type: "application/pdf" });
  const wait = async () => {};

  it("polls until the job completes", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-1", status: "accepted" }))
      .mockResolvedValueOnce(
        jsonResponse({
          job_id: "job-1",
          status: "processing",
          phase: "graph",
          graph_children_processed: 1,
          graph_children_total: 4,
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          job_id: "job-1",
          status: "completed",
          child_chunks_indexed: 4,
          graph_relationships_indexed: 2,
        }),
      );
    const progress = vi.fn();

    const result = await ingestPdf(file, progress, { fetchImpl, wait });

    expect(result.child_chunks_indexed).toBe(4);
    expect(progress).toHaveBeenCalledTimes(1);
  });

  it("honours an explicitly requested polling cap for controlled callers", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-1", status: "accepted" }))
      .mockResolvedValue(jsonResponse({ job_id: "job-1", status: "processing", phase: "graph" }));

    await expect(
      ingestPdf(file, () => {}, { fetchImpl, wait, maxAttempts: 3 }),
    ).rejects.toThrow(/still running/);
    expect(fetchImpl).toHaveBeenCalledTimes(4);
  });

  it("surfaces a failed job as an error", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ job_id: "job-1", status: "accepted" }))
      .mockResolvedValueOnce(
        jsonResponse({ job_id: "job-1", status: "failed", warnings: ["Ingestion failed: HTTPError"] }),
      );

    await expect(ingestPdf(file, () => {}, { fetchImpl, wait })).rejects.toThrow(
      "Ingestion failed: HTTPError",
    );
  });

  it("rejects an upload the API refused", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ detail: "Only PDF uploads are supported" }, false, 415));

    await expect(ingestPdf(file, () => {}, { fetchImpl, wait })).rejects.toThrow(
      "Only PDF uploads are supported",
    );
  });

  it("turns a proxy HTML upload error into a useful message", async () => {
    const fetchImpl = vi.fn(async () =>
      ({ ok: false, status: 413, text: async () => "<html>too large</html>" }) as Response,
    );

    await expect(ingestPdf(file, () => {}, { fetchImpl, wait })).rejects.toThrow(
      "larger than the server upload limit",
    );
  });
});

describe("cancelIngestion", () => {
  it("posts to the job cancellation endpoint", async () => {
    const fetchImpl = vi.fn(async (_url: URL | RequestInfo, _init?: RequestInit) =>
      jsonResponse({ job_id: "job-1", status: "cancelled", phase: "cancelled" }),
    );

    await expect(cancelIngestion("job-1", { fetchImpl })).resolves.toMatchObject({
      status: "cancelled",
    });
    const call = fetchImpl.mock.calls[0];
    expect(String(call[0])).toContain("/ingest/job-1/cancel");
    expect(call[1]).toEqual({ method: "POST" });
  });
});

describe("clearKnowledgeBase", () => {
  it("deletes the application's full knowledge base and returns removal counts", async () => {
    const fetchImpl = vi.fn(async (_url: URL | RequestInfo, _init?: RequestInit) =>
      jsonResponse({ vectors_removed: 8, relationships_removed: 3, entities_removed: 4 }),
    );

    await expect(clearKnowledgeBase({ fetchImpl })).resolves.toEqual({
      vectors_removed: 8,
      relationships_removed: 3,
      entities_removed: 4,
    });
    const call = fetchImpl.mock.calls[0];
    expect(String(call[0])).toContain("/knowledge-base");
    expect(call[1]).toEqual({ method: "DELETE" });
  });
});

describe("fetchKnowledgeBaseUsage", () => {
  it("reads the current scoped Qdrant and Neo4j record counts", async () => {
    const fetchImpl = vi.fn(async (_url: URL | RequestInfo, _init?: RequestInit) =>
      jsonResponse({
        qdrant_parent_vectors: 2,
        qdrant_child_vectors: 8,
        neo4j_entities: 4,
        neo4j_relationships: 3,
      }),
    );

    await expect(fetchKnowledgeBaseUsage({ fetchImpl })).resolves.toMatchObject({
      qdrant_child_vectors: 8,
      neo4j_relationships: 3,
    });
    expect(String(fetchImpl.mock.calls[0][0])).toContain("/knowledge-base/usage");
  });
});

describe("subgraphFromTriples", () => {
  it("deduplicates nodes and keeps one edge per triple", () => {
    const triples: GraphTriple[] = [
      {
        subject: "Microsoft",
        predicate: "ACQUIRED",
        object: "Activision Blizzard",
        source_parent_chunk_id: "p1",
        source_child_chunk_id: "c1",
        source_id: "doc",
        evidence: "Microsoft acquired Activision Blizzard.",
      },
      {
        subject: "Microsoft",
        predicate: "OWNS",
        object: "Game Pass",
        source_parent_chunk_id: "p1",
        source_child_chunk_id: "c2",
        source_id: "doc",
        evidence: "Microsoft owns Game Pass.",
      },
    ];

    const subgraph = subgraphFromTriples(triples);

    expect(subgraph.nodes.map((node) => node.id)).toEqual([
      "Microsoft",
      "Activision Blizzard",
      "Game Pass",
    ]);
    expect(subgraph.edges).toHaveLength(2);
  });
});
