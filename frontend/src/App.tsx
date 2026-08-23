import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Database,
  FileText,
  Menu,
  MessageSquare,
  Network,
  Plus,
  RotateCcw,
  Send,
  Settings,
  Trash2,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import AnswerContent from "./AnswerContent";
import GraphInspector from "./GraphInspector";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  fetchModelProviders,
  fetchKnowledgeBaseUsage,
  cancelIngestion,
  clearKnowledgeBase,
  deleteKnowledgeBaseDocument,
  fetchKnowledgeBaseDocuments,
  ingestPdf,
  type ModelProvider,
  type KnowledgeBaseUsage,
  type KnowledgeBaseDocument,
  streamChat,
  subgraphFromTriples,
} from "@/lib/api";
import type {
  ChatMessage,
  ChatTurn,
  Citation,
  Conversation,
  GraphTriple,
  ParentContext,
  Subgraph,
} from "./types";

const EMPTY_SUBGRAPH: Subgraph = { nodes: [], edges: [] };
const EMPTY_KNOWLEDGE_BASE_USAGE: KnowledgeBaseUsage = {
  qdrant_parent_vectors: 0,
  qdrant_child_vectors: 0,
  neo4j_entities: 0,
  neo4j_relationships: 0,
};
const CHAT_STORAGE_KEY = "graphrag-assessment.chat.v1";
const SUBGRAPH_STORAGE_KEY = "graphrag-assessment.subgraph.v1";
const CONVERSATIONS_STORAGE_KEY = "graphrag-assessment.conversations.v1";
const ACTIVE_CONVERSATION_STORAGE_KEY = "graphrag-assessment.active-conversation.v1";

function readStoredChat(): ChatMessage[] {
  try {
    const value = globalThis.localStorage?.getItem(CHAT_STORAGE_KEY);
    const parsed: unknown = value ? JSON.parse(value) : [];
    return Array.isArray(parsed) ? parsed.filter((item): item is ChatMessage =>
      typeof item === "object" && item !== null && "id" in item && "role" in item && "content" in item,
    ) : [];
  } catch {
    return [];
  }
}

function readStoredSubgraph(): Subgraph {
  try {
    const value = globalThis.localStorage?.getItem(SUBGRAPH_STORAGE_KEY);
    const parsed: unknown = value ? JSON.parse(value) : null;
    if (typeof parsed === "object" && parsed !== null && "nodes" in parsed && "edges" in parsed) {
      const graph = parsed as Subgraph;
      return Array.isArray(graph.nodes) && Array.isArray(graph.edges) ? graph : EMPTY_SUBGRAPH;
    }
  } catch {
    // Invalid or unavailable browser storage should never prevent the app loading.
  }
  return EMPTY_SUBGRAPH;
}

function createConversation(title = "New conversation"): Conversation {
  const now = new Date().toISOString();
  return { id: newId(), title, messages: [], subgraph: EMPTY_SUBGRAPH, createdAt: now, updatedAt: now };
}

function readStoredWorkspace(): { conversations: Conversation[]; activeConversationId: string } {
  try {
    const value = globalThis.localStorage?.getItem(CONVERSATIONS_STORAGE_KEY);
    const parsed: unknown = value ? JSON.parse(value) : null;
    if (Array.isArray(parsed)) {
      const conversations = parsed.filter((item): item is Conversation =>
        typeof item === "object" && item !== null && "id" in item && "title" in item
        && "messages" in item && Array.isArray(item.messages) && "subgraph" in item,
      );
      if (conversations.length > 0) {
        const storedActive = globalThis.localStorage?.getItem(ACTIVE_CONVERSATION_STORAGE_KEY);
        return {
          conversations,
          activeConversationId: conversations.some((conversation) => conversation.id === storedActive)
            ? storedActive!
            : conversations[0].id,
        };
      }
    }
  } catch {
    // Invalid storage should never prevent the app loading.
  }

  // Migrate the previous one-conversation format without losing existing work.
  const legacyMessages = readStoredChat();
  const legacySubgraph = readStoredSubgraph();
  const migrated = createConversation(
    legacyMessages.find((message) => message.role === "user")?.content.slice(0, 52) || "New conversation",
  );
  migrated.messages = legacyMessages;
  migrated.subgraph = legacySubgraph;
  return { conversations: [migrated], activeConversationId: migrated.id };
}

function clearStoredSession() {
  try {
    globalThis.localStorage?.removeItem(CHAT_STORAGE_KEY);
    globalThis.localStorage?.removeItem(SUBGRAPH_STORAGE_KEY);
    globalThis.localStorage?.removeItem(CONVERSATIONS_STORAGE_KEY);
    globalThis.localStorage?.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
  } catch {
    // Browser privacy settings can disable storage; the in-memory reset still works.
  }
}

function newId() {
  return globalThis.crypto?.randomUUID?.() ?? String(Date.now() + Math.random());
}

function messageTime(timestamp?: string) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export default function App() {
  const [question, setQuestion] = useState("What relationships are described in my documents?");
  const [workspace, setWorkspace] = useState(readStoredWorkspace);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [isIndexing, setIsIndexing] = useState(false);
  const [llmProvider, setLlmProvider] = useState<ModelProvider>("local");
  const [openRouterConfigured, setOpenRouterConfigured] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearingKnowledgeBase, setClearingKnowledgeBase] = useState(false);
  const [confirmClearBrowser, setConfirmClearBrowser] = useState(false);
  const [settingsNotice, setSettingsNotice] = useState("");
  const [knowledgeBaseUsage, setKnowledgeBaseUsage] = useState<KnowledgeBaseUsage | null>(null);
  const [knowledgeBaseDocuments, setKnowledgeBaseDocuments] = useState<KnowledgeBaseDocument[]>([]);
  const [deletingDocument, setDeletingDocument] = useState<string | null>(null);
  const [loadingKnowledgeBaseUsage, setLoadingKnowledgeBaseUsage] = useState(false);
  const streamRef = useRef<AbortController | null>(null);
  const ingestionRef = useRef<{
    controller: AbortController;
    jobId?: string;
  } | null>(null);
  const threadEnd = useRef<HTMLDivElement>(null);

  const activeConversation = workspace.conversations.find(
    (conversation) => conversation.id === workspace.activeConversationId,
  ) ?? workspace.conversations[0];
  const messages = activeConversation.messages;
  const subgraph = activeConversation.subgraph;

  const lastAnswer = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant"),
    [messages],
  );
  const sources = lastAnswer?.sources ?? [];
  const parents = lastAnswer?.parents ?? [];
  const triples = lastAnswer?.triples ?? [];
  const sourceCount = useMemo(
    () => new Set((lastAnswer?.sources ?? []).map((source) => source.source_id)).size,
    [lastAnswer],
  );
  const indexingProgress = useMemo(() => {
    const match = uploadStatus.match(/(\d+)\/(\d+) chunks/);
    if (!match) return null;
    const completed = Number(match[1]);
    const total = Number(match[2]);
    return total > 0 ? { completed, total, percent: Math.round((completed / total) * 100) } : null;
  }, [uploadStatus]);
  const loading = status.includes("Retrieving") || status.includes("Generating");

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => () => streamRef.current?.abort(), []);

  useEffect(() => {
    try {
      const saved = workspace.conversations.map((conversation) => ({
        ...conversation,
        messages: conversation.messages.filter((message) => message.status !== "streaming"),
      }));
      globalThis.localStorage?.setItem(CONVERSATIONS_STORAGE_KEY, JSON.stringify(saved));
      globalThis.localStorage?.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, workspace.activeConversationId);
    } catch {
      // Graph inspection remains usable in-memory if persistence is unavailable.
    }
  }, [workspace]);

  useEffect(() => {
    void fetchModelProviders()
      .then((providers) => {
        setLlmProvider(providers.default_provider);
        setOpenRouterConfigured(providers.openrouter_configured);
      })
      .catch(() => {
        // Local inference remains available even if provider metadata cannot load.
        setOpenRouterConfigured(false);
      });
  }, []);

  useEffect(() => {
    if (!settingsOpen) return;
    setLoadingKnowledgeBaseUsage(true);
    void Promise.all([fetchKnowledgeBaseUsage(), fetchKnowledgeBaseDocuments()])
      .then(([usage, documents]) => {
        setKnowledgeBaseUsage(usage);
        setKnowledgeBaseDocuments(documents);
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Could not load knowledge-base usage"))
      .finally(() => setLoadingKnowledgeBaseUsage(false));
  }, [settingsOpen]);

  function updateConversation(
    conversationId: string,
    change: Partial<Pick<Conversation, "messages" | "subgraph" | "title">>
      | ((conversation: Conversation) => Partial<Pick<Conversation, "messages" | "subgraph" | "title">>),
  ) {
    setWorkspace((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) =>
        conversation.id === conversationId
          ? {
              ...conversation,
              ...(typeof change === "function" ? change(conversation) : change),
              updatedAt: new Date().toISOString(),
            }
          : conversation,
      ),
    }));
  }

  function updateAnswer(conversationId: string, id: string, change: Partial<ChatMessage>) {
    updateConversation(conversationId, (conversation) => ({
      messages: conversation.messages.map((message) => message.id === id ? { ...message, ...change } : message),
    }));
  }

  function appendToken(conversationId: string, id: string, token: string) {
    updateConversation(conversationId, (conversation) => ({
      messages: conversation.messages.map((message) =>
        message.id === id ? { ...message, content: message.content + token } : message,
      ),
    }));
  }

  async function sendQuestion(query: string, priorMessages: ChatMessage[] = messages) {
    if (!query || loading) return;

    const conversationId = workspace.activeConversationId;

    streamRef.current?.abort();
    const controller = new AbortController();
    streamRef.current = controller;

    const history: ChatTurn[] = priorMessages
      .filter((message) => message.content.trim())
      .slice(-6)
      .map(({ role, content }) => ({ role, content }));
    const answerId = newId();
    const sentAt = new Date().toISOString();
    updateConversation(conversationId, {
      title: priorMessages.some((message) => message.role === "user") ? activeConversation.title : query.slice(0, 52),
      messages: [
        ...priorMessages,
        { id: newId(), role: "user", content: query, timestamp: sentAt },
        { id: answerId, role: "assistant", content: "", status: "streaming" },
      ],
    });
    setError("");
    setStatus("Retrieving vector and graph evidence…");

    let streamed: GraphTriple[] = [];
    let responseFailed = false;
    try {
      await streamChat(
        { query, history, llmProvider },
        {
          onSources: (value: Citation[]) => updateAnswer(conversationId, answerId, { sources: value }),
          onParents: (value: ParentContext[]) =>
            updateAnswer(conversationId, answerId, { parents: value }),
          onGraph: (value: GraphTriple[]) => {
            streamed = value;
            updateAnswer(conversationId, answerId, { triples: value });
            setStatus("Generating cited answer…");
          },
          onToken: (token: string) => appendToken(conversationId, answerId, token),
          onError: (message: string) => {
            responseFailed = true;
            setError(message);
            updateAnswer(conversationId, answerId, { status: "failed", timestamp: new Date().toISOString() });
          },
          onDone: () => setStatus(responseFailed ? "Failed" : "Complete"),
        },
        { signal: controller.signal },
      );
      if (!responseFailed) {
        updateAnswer(conversationId, answerId, { status: "complete", timestamp: new Date().toISOString() });
      }
      // The inspector must show the same graph evidence that was supplied to
      // the model, not a second broad traversal that can contain extra edges.
      updateConversation(conversationId, { subgraph: subgraphFromTriples(streamed) });
    } catch (caught) {
      if (controller.signal.aborted) return;
      setError(caught instanceof Error ? caught.message : "Chat request failed");
      updateAnswer(conversationId, answerId, { status: "failed", timestamp: new Date().toISOString() });
      setStatus("Failed");
    }
  }

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    const query = question.trim();
    if (!query) return;
    setQuestion("");
    await sendQuestion(query);
  }

  function resendFailedMessage(failedAnswerId: string) {
    const failedAnswerIndex = messages.findIndex((message) => message.id === failedAnswerId);
    if (failedAnswerIndex < 1) return;
    const failedQuestionIndex = messages
      .slice(0, failedAnswerIndex)
      .map((message) => message.role)
      .lastIndexOf("user");
    if (failedQuestionIndex < 0) return;

    const failedQuestion = messages[failedQuestionIndex];
    const priorMessages = messages.slice(0, failedQuestionIndex);
    // Replace the failed request pair rather than appending a duplicate turn.
    void sendQuestion(failedQuestion.content, priorMessages);
  }

  async function uploadPdf(file?: File) {
    if (!file) return;
    const controller = new AbortController();
    ingestionRef.current = { controller };
    setIsIndexing(true);
    setUploadStatus(`Queued ${file.name}…`);
    setError("");
    try {
      const result = await ingestPdf(file, (job) => {
        const progress = job.graph_children_total
          ? ` ${job.graph_children_processed}/${job.graph_children_total} chunks`
          : "";
        const phase = job.phase
          ? `${job.phase[0].toUpperCase()}${job.phase.slice(1)}`
          : "Indexing";
        setUploadStatus(`${phase} ${file.name}…${progress}`);
      }, {
        llmProvider,
        signal: controller.signal,
        onStarted: (job) => {
          if (ingestionRef.current?.controller === controller) {
            ingestionRef.current.jobId = job.job_id;
          }
        },
      });
      setUploadStatus(
        `Indexed ${result.child_chunks_indexed} child chunks and ${result.graph_relationships_indexed} graph relationships.`,
      );
    } catch (caught) {
      if (controller.signal.aborted) {
        setUploadStatus("Indexing cancelled.");
        return;
      }
      setError(caught instanceof Error ? caught.message : "Upload failed");
      setUploadStatus("");
    } finally {
      if (ingestionRef.current?.controller === controller) {
        ingestionRef.current = null;
      }
      setIsIndexing(false);
    }
  }

  async function cancelIndexing() {
    const active = ingestionRef.current;
    if (!active) return;
    setUploadStatus("Cancelling indexing…");
    active.controller.abort();
    if (!active.jobId) return;
    try {
      await cancelIngestion(active.jobId);
      setUploadStatus("Indexing cancelled.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not cancel indexing");
    }
  }

  async function clearAllKnowledgeBaseData() {
    setClearingKnowledgeBase(true);
    setError("");
    try {
      const result = await clearKnowledgeBase();
      clearStoredSession();
      const fresh = createConversation();
      setWorkspace({ conversations: [fresh], activeConversationId: fresh.id });
      setKnowledgeBaseUsage(EMPTY_KNOWLEDGE_BASE_USAGE);
      setKnowledgeBaseDocuments([]);
      const notice = `Knowledge base and saved browser evidence cleared: ${result.vectors_removed} vectors, ${result.relationships_removed} relationships, and ${result.entities_removed} entities removed.`;
      setUploadStatus(notice);
      setSettingsNotice(notice);
      setConfirmClear(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not clear the knowledge base");
    } finally {
      setClearingKnowledgeBase(false);
    }
  }

  async function deleteKnowledgeBasePdf(sourceId: string) {
    setDeletingDocument(sourceId);
    setError("");
    try {
      const result = await deleteKnowledgeBaseDocument(sourceId);
      const [usage, documents] = await Promise.all([fetchKnowledgeBaseUsage(), fetchKnowledgeBaseDocuments()]);
      setKnowledgeBaseUsage(usage);
      setKnowledgeBaseDocuments(documents);
      setSettingsNotice(`Removed ${sourceId}: ${result.vectors_removed} vector records and ${result.relationships_removed} graph relationships.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not remove document");
    } finally {
      setDeletingDocument(null);
    }
  }

  function clearBrowserSession() {
    clearStoredSession();
    const fresh = createConversation();
    setWorkspace({ conversations: [fresh], activeConversationId: fresh.id });
    setConfirmClearBrowser(false);
    setSettingsNotice("Saved chat history and graph evidence were cleared from this browser. Qdrant and Neo4j were not changed.");
  }

  return (
    <main className="dark min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 grid-noise" />
      <section className="relative mx-auto max-w-7xl px-4 py-5 sm:px-6 sm:py-6 md:px-8 md:py-8">
        <div className="mb-7 flex items-center justify-between border-b border-foreground/15 pb-4 text-[11px] font-medium tracking-[0.14em] text-muted-foreground uppercase sm:mb-10">
          <span>Rushikesh K</span>
          <span className="hidden items-center gap-2 sm:flex"><span className="size-1.5 rounded-full bg-primary" /> All systems local</span>
        </div>
        <header className="mb-7 flex flex-col justify-between gap-6 border-b border-foreground/15 pb-8 sm:mb-9 md:flex-row md:items-end md:pb-10">
          <div className="max-w-2xl">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="mb-5 rounded-full border-foreground/20 bg-transparent hover:bg-foreground hover:text-background"
              aria-label="Open navigation"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu />
            </Button>
            <h1 className="max-w-xl text-5xl font-semibold tracking-[-0.065em] sm:text-6xl md:text-7xl">
              Your documents.<br />
              <span className="text-primary">Grounded answers.</span>
            </h1>
            <p className="mt-5 max-w-lg text-base leading-7 text-muted-foreground md:text-lg">
              Ask, verify, and traverse your knowledge base without losing the source behind the answer.
            </p>
          </div>
          <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:flex-wrap sm:items-center">
            <ProviderSwitch
              provider={llmProvider}
              openRouterConfigured={openRouterConfigured}
              disabled={loading || isIndexing}
              onChange={setLlmProvider}
            />
            <Button
              asChild
              variant="outline"
              className="h-10 w-full rounded-md border-foreground/25 bg-transparent px-4 hover:border-primary hover:bg-primary hover:text-primary-foreground sm:w-auto"
            >
              <label className={isIndexing ? "cursor-not-allowed" : "cursor-pointer"}>
                <Upload /> Add a PDF
                <input
                  className="hidden"
                  type="file"
                  accept="application/pdf"
                  disabled={isIndexing}
                  onChange={(event) => uploadPdf(event.target.files?.[0])}
                />
              </label>
            </Button>
          </div>
        </header>

        {uploadStatus && (
          <Alert className="mb-6 overflow-hidden border-primary/35 bg-[linear-gradient(110deg,hsl(var(--primary)/0.18),hsl(var(--card))_48%,hsl(var(--primary)/0.10))] p-0 shadow-[0_14px_45px_-28px_hsl(var(--primary)/0.9)]">
            <div className="col-span-2 flex items-start gap-3 px-4 py-4 sm:px-5">
              <div className="relative mt-0.5 grid size-10 shrink-0 place-items-center rounded-md border border-primary/40 bg-primary/15 text-primary shadow-[0_0_24px_hsl(var(--primary)/0.28)]">
                <Database className="size-5" />
                {isIndexing && <span className="absolute -right-1 -top-1 size-2.5 animate-ping rounded-full bg-primary" />}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <AlertTitle className="text-base font-semibold tracking-tight">{isIndexing ? "Indexing your document" : "Knowledge base updated"}</AlertTitle>
                  <Badge className={isIndexing ? "border-primary/30 bg-primary/15 text-primary" : "border-primary/25 bg-primary/10 text-primary"} variant="outline">
                    {isIndexing ? "In progress" : "Complete"}
                  </Badge>
                </div>
                <AlertDescription className="mt-1.5 max-w-3xl break-words text-sm leading-6 text-muted-foreground">
                  {uploadStatus}
                </AlertDescription>
                {isIndexing && (
                  <div className="mt-3">
                    <div className="mb-1.5 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.13em] text-muted-foreground">
                      <span>{indexingProgress ? "Graph extraction" : "Preparing document"}</span>
                      <span>{indexingProgress ? `${indexingProgress.completed} of ${indexingProgress.total}` : "Working"}</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-foreground/10">
                      <div
                        className={`h-full rounded-full bg-primary shadow-[0_0_12px_hsl(var(--primary)/0.8)] transition-[width] duration-500 ${indexingProgress ? "" : "w-2/5 animate-pulse"}`}
                        style={indexingProgress ? { width: `${Math.max(4, indexingProgress.percent)}%` } : undefined}
                      />
                    </div>
                  </div>
                )}
              </div>
              {isIndexing && (
                <Button type="button" variant="outline" size="sm" className="mt-0.5 border-foreground/20 bg-background/35 hover:border-destructive/60 hover:bg-destructive/10 hover:text-destructive" onClick={cancelIndexing}>
                  <X /> Cancel
                </Button>
              )}
            </div>
          </Alert>
        )}
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertTitle>Request failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="mb-6 grid gap-3 sm:grid-cols-3">
          <Metric
            icon={<FileText />}
            label="Retrieved chunks"
            value={String(sources.length)}
            detail={`${sourceCount} source${sourceCount === 1 ? "" : "s"}`}
          />
          <Metric
            icon={<Network />}
            label="Graph evidence"
            value={String(triples.length)}
            detail="relationships in view"
          />
          <Metric
            icon={<WandSparkles />}
            label="Inference mode"
            value={llmProvider === "openrouter" ? "OpenRouter" : "Local"}
            detail={llmProvider === "openrouter" ? "Hosted LLMs + local retrieval" : "Ollama + local retrieval"}
          />
        </div>

        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.18fr)_minmax(340px,.82fr)]">
          <div className="space-y-6">
            <Card className="product-card border-foreground/15 bg-card/95 shadow-none">
              <CardHeader className="border-b border-foreground/12">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                  <CardTitle>Conversation</CardTitle>
                    <CardDescription>
                      Follow-up questions reuse the earlier turns for context.
                    </CardDescription>
                  </div>
                  <Badge
                    variant={status === "Complete" ? "secondary" : "outline"}
                    className={
                      status === "Complete" ? "bg-primary/15 text-primary" : ""
                    }
                  >
                    {status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 pt-6">
                <div
                  role="log"
                  aria-label="Conversation"
                  className="max-h-[60vh] space-y-4 overflow-y-auto pr-1 sm:max-h-[26rem]"
                >
                  {messages.length === 0 && (
                    <p className="py-6 text-muted-foreground">
                      Ask a question to stream a cited answer from your local
                      knowledge base.
                    </p>
                  )}
                  {messages.map((message) =>
                    message.role === "user" ? (
                      <div key={message.id} className="flex justify-end">
                        <div className="max-w-[92%] sm:max-w-[85%]">
                          <p className="break-words rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground">
                            {message.content}
                          </p>
                          <p className="mt-1 text-right text-[11px] text-muted-foreground">
                            {messageTime(message.timestamp) ? `Sent · ${messageTime(message.timestamp)}` : "Sent"}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div key={message.id} className="max-w-full sm:max-w-[92%]">
                        <article className="answer-content break-words rounded-2xl rounded-bl-sm border border-foreground/15 bg-background px-4 py-3 text-[0.98rem]">
                          {message.content ? (
                            <AnswerContent answer={message.content} />
                          ) : (
                            <div className="space-y-3 py-1">
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-[80%]" />
                            </div>
                          )}
                        </article>
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          {message.status === "streaming"
                            ? "Receiving…"
                            : message.status === "failed"
                              ? "Response failed"
                              : messageTime(message.timestamp)
                                ? `Received · ${messageTime(message.timestamp)}`
                                : "Received"}
                        </p>
                        {message.status === "failed" && (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="mt-2"
                            onClick={() => resendFailedMessage(message.id)}
                          >
                            <RotateCcw /> Resend message
                          </Button>
                        )}
                      </div>
                    ),
                  )}
                  <div ref={threadEnd} />
                </div>

                <Separator />

                <form onSubmit={submitQuestion} className="space-y-4">
                  <Textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    aria-label="Question"
                    className="min-h-24 resize-y bg-background text-base shadow-inner"
                    placeholder="Ask a question about your indexed documents…"
                  />
                  <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
                    <p className="text-xs leading-5 text-muted-foreground">
                      Source citations are included in every supported answer.
                    </p>
                    <Button
                      type="submit"
                      size="lg"
                      className="w-full sm:w-auto"
                      disabled={loading || !question.trim()}
                    >
                      {loading ? (
                        <>
                          <span className="size-3 animate-pulse rounded-full bg-current" />{" "}
                          Working…
                        </>
                      ) : (
                        <>
                          <Send /> Ask {llmProvider === "openrouter" ? "OpenRouter" : "local model"}
                        </>
                      )}
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>

            <EvidencePanel
              sources={sources}
              parents={parents}
              sourceCount={sourceCount}
            />
          </div>

          <aside className="min-w-0 self-start">
            <Card className="product-card border-foreground/15 bg-card/95 shadow-none xl:sticky xl:top-6">
              <CardHeader className="border-b border-foreground/12">
                <CardTitle className="flex items-center gap-2">
                  <Network className="size-4 text-primary" /> Graph inspector
                </CardTitle>
                <CardDescription>
                  Pan and zoom the graph evidence supplied to this answer.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <GraphInspector subgraph={subgraph} />
                <Separator className="my-5" />
                <div className="space-y-2.5">
                  {triples.length === 0 ? (
                    <p className="rounded-sm border border-dashed p-4 text-sm text-muted-foreground">
                      Graph evidence will appear here after a supported
                      question.
                    </p>
                  ) : (
                    triples.map((triple, index) => (
                      <div
                        key={`${triple.source_child_chunk_id}-${index}`}
                        className="rounded-sm border border-foreground/12 bg-background p-3.5 text-sm break-words"
                      >
                        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
                          <span className="font-medium text-primary">
                            {triple.subject}
                          </span>
                          <Badge variant="outline" className="h-5 text-[10px]">
                            {triple.predicate}
                          </Badge>
                          <span className="font-medium text-primary">
                            {triple.object}
                          </span>
                        </div>
                        <p className="mt-2 text-xs leading-5 text-muted-foreground">
                          {triple.evidence}
                        </p>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </aside>
        </div>
      </section>
      <NavigationSidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={workspace.conversations}
        activeConversationId={workspace.activeConversationId}
        disabled={loading}
        onCreateConversation={() => {
          const next = createConversation();
          setWorkspace((current) => ({ ...current, conversations: [next, ...current.conversations], activeConversationId: next.id }));
          setQuestion("");
          setStatus("Ready");
          setError("");
          setSidebarOpen(false);
        }}
        onSelectConversation={(conversationId) => {
          setWorkspace((current) => ({ ...current, activeConversationId: conversationId }));
          setStatus("Ready");
          setError("");
          setSidebarOpen(false);
        }}
        onDeleteConversation={(conversationId) => {
          setWorkspace((current) => {
            const remaining = current.conversations.filter((conversation) => conversation.id !== conversationId);
            const conversations = remaining.length > 0 ? remaining : [createConversation()];
            return {
              conversations,
              activeConversationId: current.activeConversationId === conversationId
                ? conversations[0].id
                : current.activeConversationId,
            };
          });
        }}
        onOpenSettings={() => {
          setSettingsOpen(true);
          setSidebarOpen(false);
        }}
      />
      {settingsOpen && (
        <SettingsPage
          confirmClear={confirmClear}
          clearing={clearingKnowledgeBase}
          confirmClearBrowser={confirmClearBrowser}
          notice={settingsNotice}
          indexing={isIndexing}
          usage={knowledgeBaseUsage}
          documents={knowledgeBaseDocuments}
          deletingDocument={deletingDocument}
          loadingUsage={loadingKnowledgeBaseUsage}
          onClose={() => {
            setConfirmClear(false);
            setConfirmClearBrowser(false);
            setSettingsOpen(false);
          }}
          onRequestClear={() => setConfirmClear(true)}
          onCancelClear={() => setConfirmClear(false)}
          onConfirmClear={clearAllKnowledgeBaseData}
          onRequestClearBrowser={() => setConfirmClearBrowser(true)}
          onCancelClearBrowser={() => setConfirmClearBrowser(false)}
          onConfirmClearBrowser={clearBrowserSession}
          onDeleteDocument={deleteKnowledgeBasePdf}
        />
      )}
    </main>
  );
}

function NavigationSidebar({
  open,
  onClose,
  conversations,
  activeConversationId,
  disabled,
  onCreateConversation,
  onSelectConversation,
  onDeleteConversation,
  onOpenSettings,
}: {
  open: boolean;
  onClose: () => void;
  conversations: Conversation[];
  activeConversationId: string;
  disabled: boolean;
  onCreateConversation: () => void;
  onSelectConversation: (conversationId: string) => void;
  onDeleteConversation: (conversationId: string) => void;
  onOpenSettings: () => void;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <Button
        type="button"
        aria-label="Close navigation"
        variant="ghost"
        className="absolute inset-0 h-full w-full rounded-none bg-black/60 hover:bg-black/60"
        onClick={onClose}
      />
      <aside className="relative flex h-full w-[min(20rem,calc(100vw-1rem))] flex-col border-r bg-card p-4 shadow-2xl sm:p-5">
        <div className="flex items-center justify-between">
          <span className="font-semibold">Rushikesh K</span>
          <Button type="button" variant="ghost" size="icon" aria-label="Close navigation" onClick={onClose}>
            <X />
          </Button>
        </div>
        <Separator className="my-5" />
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">Conversations</p>
          <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={onCreateConversation}>
            <Plus /> New
          </Button>
        </div>
        <div className="mt-3 min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`group flex items-center gap-1 rounded-md border p-1 transition-colors ${conversation.id === activeConversationId ? "border-primary/35 bg-primary/10" : "border-transparent hover:border-foreground/12 hover:bg-muted/60"}`}
            >
              <Button
                type="button"
                variant="ghost"
                className={`min-w-0 flex-1 justify-start gap-2 px-2 text-left ${conversation.id === activeConversationId ? "text-primary" : ""}`}
                disabled={disabled}
                onClick={() => onSelectConversation(conversation.id)}
              >
                <MessageSquare className="size-4 shrink-0" />
                <span className="truncate">{conversation.title}</span>
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8 shrink-0 text-muted-foreground opacity-100 hover:text-destructive sm:opacity-0 sm:group-hover:opacity-100"
                disabled={disabled}
                aria-label={`Delete ${conversation.title}`}
                onClick={() => onDeleteConversation(conversation.id)}
              >
                <Trash2 className="size-4" />
              </Button>
            </div>
          ))}
        </div>
        <Separator className="my-5" />
        <Button type="button" variant="secondary" className="justify-start" onClick={onOpenSettings}>
          <Settings /> Settings
        </Button>
        <p className="mt-auto text-xs leading-5 text-muted-foreground">
          Manage inference preferences and your local knowledge base.
        </p>
      </aside>
    </div>
  );
}

function SettingsPage({
  confirmClear,
  clearing,
  confirmClearBrowser,
  indexing,
  notice,
  usage,
  documents,
  deletingDocument,
  loadingUsage,
  onClose,
  onRequestClear,
  onCancelClear,
  onConfirmClear,
  onRequestClearBrowser,
  onCancelClearBrowser,
  onConfirmClearBrowser,
  onDeleteDocument,
}: {
  confirmClear: boolean;
  clearing: boolean;
  confirmClearBrowser: boolean;
  indexing: boolean;
  notice: string;
  usage: KnowledgeBaseUsage | null;
  documents: KnowledgeBaseDocument[];
  deletingDocument: string | null;
  loadingUsage: boolean;
  onClose: () => void;
  onRequestClear: () => void;
  onCancelClear: () => void;
  onConfirmClear: () => void;
  onRequestClearBrowser: () => void;
  onCancelClearBrowser: () => void;
  onConfirmClearBrowser: () => void;
  onDeleteDocument: (sourceId: string) => void;
}) {
  return (
    <div className="fixed inset-0 z-40 overflow-y-auto bg-background/95 px-4 py-5 backdrop-blur-sm sm:px-6 sm:py-8 md:px-8 md:py-12">
      <section className="mx-auto max-w-3xl">
        <div className="mb-6 flex items-start justify-between gap-3 sm:mb-8 sm:gap-4">
          <div className="min-w-0">
            <Badge variant="outline" className="mb-3 border-primary/35 bg-primary/10 text-primary">
              <Settings /> Settings
            </Badge>
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Workspace settings</h2>
            <p className="mt-2 text-muted-foreground">Manage the data stored by this local GraphRAG workspace.</p>
          </div>
          <Button type="button" variant="outline" onClick={onClose}>
            <X /> Close
          </Button>
        </div>
        <Card className="mb-6 bg-card">
          <CardHeader>
            <CardTitle>Knowledge-base records</CardTitle>
            <CardDescription>
              Current records in this workspace.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            <UsageMetric label="Qdrant parent vectors" value={usage?.qdrant_parent_vectors} loading={loadingUsage} />
            <UsageMetric label="Qdrant child vectors" value={usage?.qdrant_child_vectors} loading={loadingUsage} />
            <UsageMetric label="Neo4j entities" value={usage?.neo4j_entities} loading={loadingUsage} />
            <UsageMetric label="Neo4j relationships" value={usage?.neo4j_relationships} loading={loadingUsage} />
          </CardContent>
        </Card>
        <Card className="mb-6 bg-card">
          <CardHeader>
            <CardTitle>Indexed documents</CardTitle>
            <CardDescription>Remove one PDF without affecting the rest of the knowledge base.</CardDescription>
          </CardHeader>
          <CardContent>
            {loadingUsage ? <Skeleton className="h-12 w-full" /> : documents.length === 0 ? (
              <p className="text-sm text-muted-foreground">No indexed PDFs found.</p>
            ) : (
              <div className="space-y-2">
                {documents.map((document) => (
                  <div key={document.source_id} className="flex flex-col gap-3 rounded-lg border bg-background/55 p-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="min-w-0">
                      <p className="truncate font-medium" title={document.source_id}>{document.source_id}</p>
                      <p className="text-xs text-muted-foreground">{document.parent_vectors} parent · {document.child_vectors} child vectors · {document.providers.join(" + ")}</p>
                    </div>
                    <Button type="button" variant="outline" size="sm" className="shrink-0 text-destructive hover:text-destructive" disabled={indexing || deletingDocument !== null} onClick={() => {
                      if (window.confirm(`Remove ${document.source_id} from the knowledge base?`)) onDeleteDocument(document.source_id);
                    }}>
                      <Trash2 /> {deletingDocument === document.source_id ? "Removing…" : "Remove PDF"}
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
        {notice && (
          <Alert className="mb-6 border-primary/30 bg-primary/10">
            <Database className="text-primary" />
            <AlertTitle>Storage updated</AlertTitle>
            <AlertDescription>{notice}</AlertDescription>
          </Alert>
        )}
        <Card className="mb-6 bg-card">
          <CardHeader>
            <CardTitle>Saved chat on this device</CardTitle>
            <CardDescription>
              Your conversations and graph view are saved here so they remain after a refresh. Clearing them does not delete your indexed documents or knowledge graph.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {confirmClearBrowser ? (
              <Alert>
                <AlertTitle>Clear saved browser history?</AlertTitle>
                <AlertDescription className="mt-3 flex flex-wrap gap-3">
                  <Button type="button" variant="outline" onClick={onConfirmClearBrowser}>
                    Clear browser history
                  </Button>
                  <Button type="button" variant="ghost" onClick={onCancelClearBrowser}>Cancel</Button>
                </AlertDescription>
              </Alert>
            ) : (
              <Button type="button" variant="outline" onClick={onRequestClearBrowser}>
                Clear saved chats and graph view
              </Button>
            )}
          </CardContent>
        </Card>
        <Card className="border-destructive/40 bg-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive"><Trash2 className="size-4" /> Danger zone</CardTitle>
            <CardDescription>
              Clear all document embeddings in Qdrant and all extracted entities and relationships in Neo4j.
              This cannot be undone.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {confirmClear ? (
              <Alert variant="destructive">
                <AlertTitle>Clear the entire knowledge base?</AlertTitle>
                <AlertDescription className="mt-3 flex flex-wrap gap-3">
                  <Button type="button" variant="destructive" disabled={clearing} onClick={onConfirmClear}>
                    <Trash2 /> {clearing ? "Clearing…" : "Yes, clear all data"}
                  </Button>
                  <Button type="button" variant="outline" disabled={clearing} onClick={onCancelClear}>Cancel</Button>
                </AlertDescription>
              </Alert>
            ) : (
              <Button type="button" variant="destructive" disabled={indexing} onClick={onRequestClear}>
                <Trash2 /> Clear knowledge base
              </Button>
            )}
            {indexing && <p className="text-sm text-muted-foreground">Cancel or wait for indexing to finish before clearing stored data.</p>}
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function UsageMetric({ label, value, loading }: { label: string; value?: number; loading: boolean }) {
  return (
    <div className="rounded-lg border bg-background/55 p-4">
      <p className="text-sm leading-5 text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{loading ? "…" : (value ?? "—")}</p>
    </div>
  );
}

function ProviderSwitch({
  provider,
  openRouterConfigured,
  disabled,
  onChange,
}: {
  provider: ModelProvider;
  openRouterConfigured: boolean;
  disabled: boolean;
  onChange: (provider: ModelProvider) => void;
}) {
  const usingOpenRouter = provider === "openrouter";
  return (
    <div className="flex w-full items-center justify-between gap-2 rounded-md border bg-card/70 px-3 py-2 shadow-sm sm:w-auto sm:justify-start sm:gap-2.5">
      <span className="pl-2 text-xs font-medium text-muted-foreground">LLM</span>
      <span className={usingOpenRouter ? "text-xs text-muted-foreground" : "text-xs font-medium text-primary"}>
        Local
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={usingOpenRouter}
        aria-label="Use OpenRouter models"
        disabled={disabled || !openRouterConfigured}
        onClick={() => onChange(usingOpenRouter ? "local" : "openrouter")}
        className="relative h-6 w-11 shrink-0 rounded-full border border-border bg-muted outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
        title={openRouterConfigured ? "Switch between local Ollama and OpenRouter models" : "Add OPENROUTER_API_KEY to enable OpenRouter"}
      >
        <span
          className={`absolute inset-y-0.5 left-0.5 size-4 rounded-full bg-foreground shadow-sm transition-transform ${usingOpenRouter ? "translate-x-5 bg-primary" : "translate-x-0"}`}
        />
      </button>
      <span className={usingOpenRouter ? "text-xs font-medium text-primary" : "text-xs text-muted-foreground"}>
        OpenRouter
      </span>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  detail,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <Card size="sm" className="metric-card border-foreground/15 bg-card/85 py-3 shadow-none">
      <CardContent className="flex items-center gap-3">
        <span className="grid size-9 place-items-center rounded-sm bg-primary/12 text-primary">
          {icon}
        </span>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="font-medium">
            {value}{" "}
            <span className="font-normal text-muted-foreground">
              · {detail}
            </span>
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function EvidencePanel({
  sources,
  parents,
  sourceCount,
}: {
  sources: Citation[];
  parents: ParentContext[];
  sourceCount: number;
}) {
  return (
    <Card className="product-card border-foreground/15 bg-card/95 shadow-none">
      <CardHeader>
        <CardTitle>Evidence trail</CardTitle>
        <CardDescription>
          Open a citation to inspect the exact retrieved excerpt and parent
          context.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <details
          className="group rounded-sm border bg-background/45 p-4"
          open={sources.length > 0}
        >
          <summary className="cursor-pointer list-none font-medium">
            <span className="flex items-center justify-between gap-3">
              <span className="min-w-0">Retrieved citations</span>
              <Badge variant="secondary">
                {sources.length} chunks · {sourceCount} sources
              </Badge>
            </span>
          </summary>
          <div className="mt-4 space-y-2">
            {sources.length ? (
              sources.map((source, index) => (
                <details
                  key={source.child_chunk_id ?? index}
                  className="rounded-sm border bg-card p-3"
                >
                  <summary className="cursor-pointer break-words text-sm font-medium text-primary">
                    S{index + 1} · {source.source_id}
                  </summary>
                  <p className="mt-2 text-sm leading-6 text-muted-foreground">
                    {source.excerpt}
                  </p>
                </details>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">
                No citations retrieved yet.
              </p>
            )}
          </div>
        </details>
        <details className="rounded-sm border bg-background/45 p-4">
          <summary className="cursor-pointer list-none font-medium">
            Parent context{" "}
            <span className="ml-1 text-muted-foreground">
              · {parents.length} blocks
            </span>
          </summary>
          <div className="mt-4 space-y-3">
            {parents.map((parent) => (
              <div
                key={parent.parent_chunk_id}
                className="rounded-sm border bg-card p-3"
              >
                <p className="mb-2 text-xs font-medium text-primary">
                  {parent.source_id}
                </p>
                <p className="text-sm leading-6 text-muted-foreground">
                  {parent.text}
                </p>
              </div>
            ))}
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
