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
}
