import type { ReactNode } from 'react'

function inline(text: string): ReactNode[] {
  const tokens = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\[(?:S|G)\d+(?:\s*,\s*(?:S|G)\d+)*\])/g)
  return tokens.filter(Boolean).map((token, index) => {
    if (token.startsWith('**') && token.endsWith('**')) return <strong key={index}>{token.slice(2, -2)}</strong>
    if (token.startsWith('*') && token.endsWith('*')) return <em key={index}>{token.slice(1, -1)}</em>
    if (/^\[(?:S|G)\d+/.test(token)) return <span className="citation-chip" key={index}>{token}</span>
    return <span key={index}>{token}</span>
  })
}

/** A deliberately small, safe Markdown renderer for model output. */
export default function AnswerContent({ answer }: { answer: string }) {
  const blocks: ReactNode[] = []
  const lines = answer.split('\n')
  let list: string[] = []

  function flushList() {
    if (!list.length) return
    blocks.push(<ul className="answer-list" key={`list-${blocks.length}`}>{list.map((item, index) => <li key={index}>{inline(item)}</li>)}</ul>)
    list = []
  }

  lines.forEach((raw, index) => {
    const line = raw.trim()
    const bullet = line.match(/^[-*]\s+(.+)$/)
    if (bullet) { list.push(bullet[1]); return }
    flushList()
    if (!line) return
    const heading = line.match(/^#{1,3}\s+(.+)$/)
    if (heading) { blocks.push(<h3 key={`heading-${index}`} className="answer-heading">{inline(heading[1])}</h3>); return }
    blocks.push(<p key={`paragraph-${index}`} className="answer-paragraph">{inline(line)}</p>)
  })
  flushList()
  return <>{blocks}</>
}
