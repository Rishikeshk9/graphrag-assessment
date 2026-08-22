import { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'
import type { GraphTriple } from './types'

export default function GraphInspector({ triples }: { triples: GraphTriple[] }) {
  const host = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!host.current) return
    const nodes = [...new Set(triples.flatMap((triple) => [triple.subject, triple.object]))]
      .map((name) => ({ data: { id: name, label: name } }))
    const edges = triples.map((triple, index) => ({
      data: { id: `${index}-${triple.source_child_chunk_id}`, source: triple.subject, target: triple.object, label: triple.predicate },
    }))
    const graph = cytoscape({
      container: host.current,
      elements: [...nodes, ...edges],
      style: [
        { selector: 'node', style: { 'background-color': '#2dd4bf', label: 'data(label)', color: '#e2e8f0', 'font-size': 11, 'text-valign': 'bottom', 'text-margin-y': 7, width: 26, height: 26 } },
        { selector: 'edge', style: { width: 2, 'line-color': '#64748b', 'target-arrow-color': '#64748b', 'target-arrow-shape': 'triangle', label: 'data(label)', color: '#94a3b8', 'font-size': 9, 'text-rotation': 'autorotate', 'curve-style': 'bezier' } },
      ],
      layout: { name: 'cose', animate: false, padding: 28 },
    })
    return () => graph.destroy()
  }, [triples])

  return <div ref={host} className="h-72 w-full rounded-xl border border-slate-700 bg-slate-950" />
}
