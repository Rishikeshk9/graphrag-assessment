import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "@/App";

// Cytoscape needs a canvas renderer that jsdom does not provide; the inspector
// is exercised through its own subgraph input instead.
vi.mock("cytoscape", () => ({ default: () => ({ destroy: () => {} }) }));

function sseStream(frames: string[]) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
}

function answerFor(text: string) {
  return sseStream([
    'event: sources\ndata: [{"parent_chunk_id":"p1","child_chunk_id":"c1","source_id":"microsoft-email","excerpt":"Microsoft acquired Activision Blizzard."}]\n\n',
    'event: parents\ndata: [{"parent_chunk_id":"p1","source_id":"microsoft-email","text":"Microsoft acquired Activision Blizzard.","matching_child_chunk_ids":["c1"]}]\n\n',
    'event: graph\ndata: [{"subject":"Microsoft","predicate":"ACQUIRED","object":"Activision Blizzard","source_parent_chunk_id":"p1","source_child_chunk_id":"c1","source_id":"microsoft-email","evidence":"Microsoft acquired Activision Blizzard."}]\n\n',
    `event: token\ndata: ${JSON.stringify(text)}\n\n`,
    "event: done\ndata: {}\n\n",
  ]);
}

const chatBodies: string[] = [];

beforeEach(() => {
  chatBodies.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, init?: RequestInit) => {
      const target = String(url);
      if (target.includes("/health/model-providers")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            default_provider: "local",
            openrouter_configured: true,
            embedding_provider: "local",
          }),
        } as unknown as Response;
      }
      if (target.includes("/chat")) {
        chatBodies.push(String(init?.body));
        const turn = chatBodies.length === 1 ? "Microsoft did [S1]." : "It paid $68.7B [S1].";
        return { ok: true, status: 200, body: answerFor(turn) } as unknown as Response;
      }
      if (target.includes("/knowledge-base")) {
        if (target.includes("/usage")) {
          return {
            ok: true,
            status: 200,
            text: async () => JSON.stringify({
              qdrant_parent_vectors: 2,
              qdrant_child_vectors: 8,
              neo4j_entities: 4,
              neo4j_relationships: 3,
            }),
          } as unknown as Response;
        }
        return {
          ok: true,
          status: 200,
          text: async () => JSON.stringify({
            vectors_removed: 8,
            relationships_removed: 3,
            entities_removed: 4,
          }),
        } as unknown as Response;
      }
      throw new Error(`unexpected request: ${target}`);
    }),
  );
});

afterEach(() => vi.unstubAllGlobals());

async function ask(question: string) {
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: question } });
  fireEvent.click(screen.getByRole("button", { name: /ask (local model|openrouter)/i }));
}

it("keeps the thread and sends prior turns as history on a follow-up", async () => {
  render(<App />);

  await ask("Who acquired Activision Blizzard?");
  await waitFor(() => expect(screen.getByText(/Microsoft did/)).toBeInTheDocument());

  await ask("What did it pay?");
  await waitFor(() => expect(screen.getByText(/It paid/)).toBeInTheDocument());

  expect(screen.getByText("Who acquired Activision Blizzard?")).toBeInTheDocument();
  expect(JSON.parse(chatBodies[0]).history).toEqual([]);
  expect(JSON.parse(chatBodies[1]).history).toEqual([
    { role: "user", content: "Who acquired Activision Blizzard?" },
    { role: "assistant", content: "Microsoft did [S1]." },
  ]);
});

it("renders streamed citations and inspects the graph evidence sent to the answer", async () => {
  render(<App />);

  await ask("Who acquired Activision Blizzard?");

  await waitFor(() => expect(screen.getByText("ACQUIRED")).toBeInTheDocument());
  expect(screen.getByText(/S1 · microsoft-email/)).toBeInTheDocument();
  expect(screen.getByText("ACQUIRED")).toBeInTheDocument();
});

it("sends the selected OpenRouter provider with chat requests", async () => {
  render(<App />);

  const providerSwitch = await screen.findByRole("switch", { name: /use openrouter models/i });
  fireEvent.click(providerSwitch);
  await ask("Who acquired Activision Blizzard?");

  await waitFor(() => expect(screen.getByText(/Microsoft did/)).toBeInTheDocument());
  expect(JSON.parse(chatBodies[0]).llm_provider).toBe("openrouter");
});

it("requires a confirmation before clearing stored vectors and graph evidence", async () => {
  render(<App />);

  fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
  fireEvent.click(screen.getByRole("button", { name: /settings/i }));
  expect(screen.getByRole("heading", { name: "Workspace settings" })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Clear knowledge base" }));
  expect(screen.getByText("Clear the entire knowledge base?")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Yes, clear all data" }));

  await waitFor(() =>
    expect(screen.getAllByText(/Knowledge base and saved browser evidence cleared: 8 vectors, 3 relationships, and 4 entities removed/)).not.toHaveLength(0),
  );
});

it("persists a completed chat and can clear browser storage without touching the API", async () => {
  const first = render(<App />);
  await ask("Who acquired Activision Blizzard?");
  await waitFor(() => expect(screen.getByText(/Microsoft did/)).toBeInTheDocument());
  first.unmount();

  render(<App />);
  expect(screen.getByText(/Microsoft did/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
  fireEvent.click(screen.getByRole("button", { name: /settings/i }));
  fireEvent.click(screen.getByRole("button", { name: "Clear saved chats and graph view" }));
  expect(screen.getByText("Clear saved browser history?")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Clear browser history" }));

  await waitFor(() => expect(screen.getByText(/saved chat history and graph evidence were cleared/i)).toBeInTheDocument());
  expect(localStorage.getItem("graphrag-assessment.chat.v1")).toBeNull();
});
