import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import type { Subgraph } from "./types";

export default function GraphInspector({ subgraph }: { subgraph: Subgraph }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!host.current) return;
    const elements = [
      ...subgraph.nodes.map((node) => ({
        data: { id: node.id, label: node.label, type: node.type ?? "Entity" },
      })),
      ...subgraph.edges.map((edge) => ({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
        },
      })),
    ];
    const graph = cytoscape({
      container: host.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#36e6aa",
            label: "data(label)",
            color: "#f5f5ef",
            "font-size": 11,
            "text-valign": "bottom",
            "text-margin-y": 7,
            width: 26,
            height: 26,
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#738078",
            "target-arrow-color": "#738078",
            "target-arrow-shape": "triangle",
            label: "data(label)",
            color: "#aeb8b2",
            "font-size": 9,
            "text-rotation": "autorotate",
            "curve-style": "bezier",
          },
        },
      ],
      layout: { name: "cose", animate: false, padding: 28 },
    });
    return () => graph.destroy();
  }, [subgraph]);

  return (
    <div
      ref={host}
      aria-label="Knowledge graph inspector"
      className="h-56 w-full touch-pan-y rounded-md border border-foreground/15 bg-black/45 sm:h-72"
    />
  );
}
