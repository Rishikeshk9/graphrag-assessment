export type Citation = {
  parent_chunk_id: string
  child_chunk_id: string | null
  source_id: string
  excerpt: string
}

export type ParentContext = {
  parent_chunk_id: string
  source_id: string
  text: string
  matching_child_chunk_ids: string[]
}

export type GraphTriple = {
  subject: string
  predicate: string
  object: string
  source_parent_chunk_id: string
  source_child_chunk_id: string
  source_id: string
  evidence: string
  /** Every document supporting this fact, not just the primary one. */
  source_ids?: string[]
  supporting_child_chunk_ids?: string[]
}

export type GraphNode = {
  id: string
  label: string
  type?: string
}

export type GraphEdge = {
  id: string
  source: string
  target: string
  label: string
  evidence?: string
  source_id?: string
  parent_chunk_id?: string
  child_chunk_id?: string
  supporting_sources?: string
  supporting_chunks?: string
}

export type Subgraph = {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export type ChatTurn = {
  role: 'user' | 'assistant'
  content: string
}

export type ChatMessage = ChatTurn & {
  id: string
  /** Local browser time: sent for user turns, received for assistant turns. */
  timestamp?: string
  sources?: Citation[]
  parents?: ParentContext[]
  triples?: GraphTriple[]
  status?: 'streaming' | 'complete' | 'failed'
}
