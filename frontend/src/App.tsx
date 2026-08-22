import { FormEvent, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { Database, FileText, Network, Send, Sparkles, Upload, WandSparkles } from 'lucide-react'
import AnswerContent from './AnswerContent'
import GraphInspector from './GraphInspector'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import type { Citation, GraphTriple, ParentContext } from './types'

const API = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8081/api/v1'

function parseSse(frame: string) {
  const event = frame.match(/^event: (.+)$/m)?.[1]
  const raw = frame.match(/^data: (.*)$/m)?.[1]
  return event && raw ? { event, data: JSON.parse(raw) } : null
}

export default function App() {
  const [question, setQuestion] = useState('Who acquired Activision Blizzard?')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Citation[]>([])
  const [parents, setParents] = useState<ParentContext[]>([])
  const [triples, setTriples] = useState<GraphTriple[]>([])
  const [status, setStatus] = useState('Ready')
  const [error, setError] = useState('')
  const [uploadStatus, setUploadStatus] = useState('')

  const sourceCount = useMemo(() => new Set(sources.map((source) => source.source_id)).size, [sources])

  async function submitQuestion(event: FormEvent) {
    event.preventDefault()
    if (!question.trim()) return
    setAnswer(''); setSources([]); setParents([]); setTriples([]); setError(''); setStatus('Retrieving vector and graph evidence…')
    try {
      const response = await fetch(`${API}/chat`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ query: question, graph_hops: 2 }) })
      if (!response.ok || !response.body) throw new Error(`Chat request failed (${response.status})`)
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
        const frames = buffer.split('\n\n'); buffer = frames.pop() ?? ''
        for (const frame of frames) {
          const parsed = parseSse(frame); if (!parsed) continue
          if (parsed.event === 'sources') setSources(parsed.data)
          if (parsed.event === 'parents') setParents(parsed.data)
          if (parsed.event === 'graph') { setTriples(parsed.data); setStatus('Generating cited answer…') }
          if (parsed.event === 'token') setAnswer((current) => current + parsed.data)
          if (parsed.event === 'error') setError(parsed.data)
          if (parsed.event === 'done') setStatus('Complete')
        }
        if (done) break
      }
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Chat request failed'); setStatus('Failed') }
  }

  async function uploadPdf(file?: File) {
    if (!file) return
    setUploadStatus(`Queued ${file.name}…`); setError('')
    try {
      const form = new FormData(); form.append('file', file)
      const response = await fetch(`${API}/ingest/file`, { method: 'POST', body: form })
      const job = await response.json()
      if (!response.ok) throw new Error(job.detail ?? 'Upload failed')
      while (true) {
        const statusResponse = await fetch(`${API}/ingest/${job.job_id}`)
        const result = await statusResponse.json()
        if (!statusResponse.ok) throw new Error(result.detail ?? 'Could not read indexing progress')
        if (result.status === 'completed') {
          setUploadStatus(`Indexed ${result.child_chunks_indexed} child chunks and ${result.graph_relationships_indexed} graph relationships.`)
          break
        }
        if (result.status === 'failed') throw new Error(result.warnings?.[0] ?? 'Indexing failed')
        const progress = result.graph_children_total ? ` ${result.graph_children_processed}/${result.graph_children_total} chunks` : ''
        setUploadStatus(`${result.phase[0].toUpperCase()}${result.phase.slice(1)} ${file.name}…${progress}`)
        await new Promise((resolve) => window.setTimeout(resolve, 1200))
      }
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Upload failed'); setUploadStatus('') }
  }

  const loading = status.includes('Retrieving') || status.includes('Generating')

  return <main className="dark min-h-screen bg-background text-foreground">
    <div className="pointer-events-none fixed inset-x-0 top-0 h-100 bg-[radial-gradient(ellipse_at_top,oklch(0.3_0.08_248/.45),transparent_65%)]" />
    <section className="relative mx-auto max-w-7xl px-5 py-8 md:px-8 md:py-12">
      <header className="mb-8 flex flex-col justify-between gap-6 border-b pb-8 md:flex-row md:items-end">
        <div className="max-w-2xl"><Badge variant="outline" className="mb-4 border-primary/35 bg-primary/10 px-2.5 text-primary"><Sparkles /> Local-first GraphRAG</Badge><h1 className="text-3xl font-semibold tracking-tight md:text-5xl">Evidence Studio</h1><p className="mt-4 text-base leading-7 text-muted-foreground md:text-lg">Ask questions across your documents, inspect every source, and follow entity relationships through the knowledge graph.</p></div>
        <Button asChild variant="outline" className="h-10 border-dashed bg-card/60 px-4 hover:bg-accent"><label className="cursor-pointer"><Upload /> Add a PDF<input className="hidden" type="file" accept="application/pdf" onChange={(event) => uploadPdf(event.target.files?.[0])} /></label></Button>
      </header>

      {uploadStatus && <Alert className="mb-6 border-primary/25 bg-primary/8"><Database className="text-primary" /><AlertTitle>Knowledge-base update</AlertTitle><AlertDescription>{uploadStatus}</AlertDescription></Alert>}
      {error && <Alert variant="destructive" className="mb-6"><AlertTitle>Request failed</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>}

      <div className="mb-6 grid gap-3 sm:grid-cols-3">
        <Metric icon={<FileText />} label="Retrieved chunks" value={String(sources.length)} detail={`${sourceCount} source${sourceCount === 1 ? '' : 's'}`} />
        <Metric icon={<Network />} label="Graph evidence" value={String(triples.length)} detail="relationships in view" />
        <Metric icon={<WandSparkles />} label="Inference mode" value="Local" detail="Ollama + Qdrant + Neo4j" />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.18fr)_minmax(340px,.82fr)]">
        <div className="space-y-6">
          <Card className="border bg-card/90 shadow-2xl shadow-black/15">
            <CardHeader><CardTitle>Ask your knowledge graph</CardTitle><CardDescription>Answers are grounded in retrieved document context and graph evidence.</CardDescription></CardHeader>
            <CardContent>
              <form onSubmit={submitQuestion} className="space-y-4"><Textarea value={question} onChange={(event) => setQuestion(event.target.value)} className="min-h-30 resize-y bg-background text-base shadow-inner" placeholder="Ask a question about your indexed documents…" /><div className="flex flex-wrap items-center justify-between gap-3"><p className="text-xs text-muted-foreground">Source citations are included in every supported answer.</p><Button type="submit" size="lg" disabled={loading || !question.trim()}>{loading ? <><span className="size-3 animate-pulse rounded-full bg-current" /> Working…</> : <><Send /> Ask local model</>}</Button></div></form>
            </CardContent>
          </Card>

          <Card className="border bg-card/90">
            <CardHeader className="border-b"><div className="flex items-center justify-between gap-4"><div><CardTitle>Grounded answer</CardTitle><CardDescription>Generated only after retrieval and graph traversal.</CardDescription></div><Badge variant={status === 'Complete' ? 'secondary' : 'outline'} className={status === 'Complete' ? 'bg-primary/15 text-primary' : ''}>{status}</Badge></div></CardHeader>
            <CardContent className="pt-6"><article className="answer-content text-[0.98rem] text-foreground md:text-[1.03rem]">{answer ? <AnswerContent answer={answer} /> : loading ? <div className="space-y-3"><Skeleton className="h-5 w-full" /><Skeleton className="h-5 w-[91%]" /><Skeleton className="h-5 w-[70%]" /></div> : <div className="py-7 text-muted-foreground">Ask a question to stream a cited answer from your local knowledge base.</div>}</article></CardContent>
          </Card>

          <EvidencePanel sources={sources} parents={parents} sourceCount={sourceCount} />
        </div>

        <aside><Card className="sticky top-6 border bg-card/90"><CardHeader><CardTitle className="flex items-center gap-2"><Network className="size-4 text-primary" /> Graph inspector</CardTitle><CardDescription>Pan and zoom the relationships used for this answer.</CardDescription></CardHeader><CardContent><GraphInspector triples={triples} /><Separator className="my-5" /><div className="space-y-2.5">{triples.length === 0 ? <p className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">Graph evidence will appear here after a supported question.</p> : triples.map((triple, index) => <div key={`${triple.source_child_chunk_id}-${index}`} className="rounded-lg border bg-background/55 p-3.5 text-sm"><div className="flex flex-wrap items-center gap-x-1.5 gap-y-1"><span className="font-medium text-primary">{triple.subject}</span><Badge variant="outline" className="h-5 text-[10px]">{triple.predicate}</Badge><span className="font-medium text-primary">{triple.object}</span></div><p className="mt-2 text-xs leading-5 text-muted-foreground">{triple.evidence}</p></div>)}</div></CardContent></Card></aside>
      </div>
    </section>
  </main>
}

function Metric({ icon, label, value, detail }: { icon: ReactNode, label: string, value: string, detail: string }) {
  return <Card size="sm" className="border bg-card/70 py-3"><CardContent className="flex items-center gap-3"><span className="grid size-9 place-items-center rounded-lg bg-primary/12 text-primary">{icon}</span><div><p className="text-xs text-muted-foreground">{label}</p><p className="font-medium">{value} <span className="font-normal text-muted-foreground">· {detail}</span></p></div></CardContent></Card>
}

function EvidencePanel({ sources, parents, sourceCount }: { sources: Citation[], parents: ParentContext[], sourceCount: number }) {
  return <Card className="border bg-card/90"><CardHeader><CardTitle>Evidence trail</CardTitle><CardDescription>Open a citation to inspect the exact retrieved excerpt and parent context.</CardDescription></CardHeader><CardContent className="space-y-3"><details className="group rounded-lg border bg-background/45 p-4" open={sources.length > 0}><summary className="cursor-pointer list-none font-medium"><span className="flex items-center justify-between"><span>Retrieved citations</span><Badge variant="secondary">{sources.length} chunks · {sourceCount} sources</Badge></span></summary><div className="mt-4 space-y-2">{sources.length ? sources.map((source, index) => <details key={source.child_chunk_id ?? index} className="rounded-md border bg-card p-3"><summary className="cursor-pointer text-sm font-medium text-primary">S{index + 1} · {source.source_id}</summary><p className="mt-2 text-sm leading-6 text-muted-foreground">{source.excerpt}</p></details>) : <p className="text-sm text-muted-foreground">No citations retrieved yet.</p>}</div></details><details className="rounded-lg border bg-background/45 p-4"><summary className="cursor-pointer list-none font-medium">Parent context <span className="ml-1 text-muted-foreground">· {parents.length} blocks</span></summary><div className="mt-4 space-y-3">{parents.map((parent) => <div key={parent.parent_chunk_id} className="rounded-md border bg-card p-3"><p className="mb-2 text-xs font-medium text-primary">{parent.source_id}</p><p className="text-sm leading-6 text-muted-foreground">{parent.text}</p></div>)}</div></details></CardContent></Card>
}
