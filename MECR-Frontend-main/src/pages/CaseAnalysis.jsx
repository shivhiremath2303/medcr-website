import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import ReactFlow, {
  Background, BaseEdge, Controls, Handle, MarkerType, MiniMap,
  Panel, Position, getBezierPath, getSmoothStepPath
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  getAnomalies, getCase, getConflictMatrix, getDocumentAnalysis, getEntities,
  getGraph, getReportContent, runAnalysis
} from '../lib/api'
import { Badge, Button, Card, Eyebrow } from '../components/ui/Primitives'

const TABS = [
  { id: 'matrix', label: 'Conflicts' },
  { id: 'anomalies', label: 'Anomalies' },
  { id: 'viewer', label: 'Documents' },
  { id: 'graph', label: 'Knowledge Graph' },
  { id: 'entities', label: 'Entities (NER)' },
  { id: 'report', label: 'Report' }
]

const ENTITY_STYLES = {
  COURT: 'bg-courtsoft text-court',
  JUDGE: 'bg-courtsoft text-court',
  PLACE: 'bg-courtsoft text-court',
  LAWYER: 'bg-goldsoft text-[#8A6A20]',
  EVIDENCE: 'bg-goldsoft text-[#8A6A20]',
  DEFENDANT: 'bg-conflictsoft text-conflict',
  EVENT: 'bg-conflictsoft text-conflict',
  WITNESS: 'bg-verifiedsoft text-verified',
  DATE: 'bg-verifiedsoft text-verified'
}

const SEVERITY_TONE = { material: 'conflict', low: 'gold', high: 'conflict', medium: 'gold' }
const ANOMALY_TONE = { contradiction: 'conflict', missing_information: 'gold', manipulation: 'conflict' }

function Metric({ label, value }) {
  return (
    <div className="bg-paperdim/50 rounded-md px-3 py-2.5">
      <div className="text-[10.5px] font-mono uppercase text-textsoft">{label}</div>
      <div className="font-serif text-xl font-semibold text-ink">{value}</div>
    </div>
  )
}

export default function CaseAnalysis() {
  const { caseId } = useParams()
  const [tab, setTab] = useState('matrix')
  const [caseInfo, setCaseInfo] = useState(null)
  const [notFound, setNotFound] = useState(false)
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState('')

  useEffect(() => {
    setNotFound(false)
    getCase(caseId).then(setCaseInfo).catch(() => setNotFound(true))
  }, [caseId])

  async function handleRerun() {
    setRunning(true)
    try {
      await runAnalysis(caseId, setStage)
      setCaseInfo(await getCase(caseId))
    } catch (err) {
      setNotFound(true)
    } finally {
      setRunning(false)
    }
  }

  if (notFound) {
    return (
      <Card className="p-8 max-w-xl">
        <Eyebrow>Case unavailable</Eyebrow>
        <h1 className="font-serif text-2xl text-ink mb-2">Case not found</h1>
        <p className="text-sm text-textsoft mb-5">
          This case does not exist or belongs to another account. Case files are
          private to the user who created them.
        </p>
        <Link to="/" className="text-sm font-semibold text-gold hover:underline">
          ← Back to my cases
        </Link>
      </Card>
    )
  }

  return (
    <div>
      <Eyebrow>Case {caseInfo?.caseNumber || caseId}</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">Analysis Workspace</h1>
      <p className="text-textsoft text-sm mb-4 max-w-lg">Everything the multi-model reasoning engine surfaced for this case.</p>

      <div className="flex items-center gap-3 mb-6 no-print">
        <button
          onClick={handleRerun}
          disabled={running}
          className="text-xs font-mono font-semibold px-3 py-1.5 rounded-md border border-line text-textsoft hover:bg-paperdim disabled:opacity-50"
        >
          {running ? 'Running…' : '↻ Re-run pipeline'}
        </button>
        {running && <span className="text-xs text-textsoft font-mono">{stage}</span>}
      </div>

      <ModelRunStrip caseId={caseId} />

      <div className="flex gap-1 border-b border-line mb-6 no-print">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-sm px-4 py-2.5 border-b-2 -mb-px transition-colors ${
              tab === t.id ? 'border-gold text-ink font-semibold' : 'border-transparent text-textsoft hover:text-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'matrix' && <ConflictList caseId={caseId} />}
      {tab === 'anomalies' && <AnomalyList caseId={caseId} />}
      {tab === 'viewer' && <Documents caseId={caseId} />}
      {tab === 'graph' && <KnowledgeGraph caseId={caseId} />}
      {tab === 'entities' && <Entities caseId={caseId} />}
      {tab === 'report' && <Report caseId={caseId} />}
    </div>
  )
}

// --- Model activity strip: proves which engines ran on this case ------------

function ModelRunStrip({ caseId }) {
  const [state, setState] = useState(null)

  useEffect(() => {
    let alive = true
    Promise.all([
      getConflictMatrix(caseId).catch(() => []),
      getAnomalies(caseId).catch(() => []),
      getEntities(caseId).catch(() => []),
      getGraph(caseId).catch(() => null),
    ]).then(([conflicts, anomalies, entities, graph]) => {
      if (alive) setState({ conflicts, anomalies, entities, graph })
    })
    return () => { alive = false }
  }, [caseId])

  if (!state) {
    return <div className="mb-6 h-12 rounded-lg border border-line bg-paperdim/40 animate-pulse" />
  }

  const conflicts = state.conflicts || []
  const anomalies = state.anomalies || []
  const legalbert = conflicts.filter((c) => c.model === 'LegalBERT').length
  const structural = conflicts.length - legalbert
  const xgb = anomalies.filter((a) => a.model === 'XGBoost').length
  const meta = state.graph?.meta || {}
  const integrity = meta.integrity || 'unscored'
  const confidence = Math.round((meta.confidence || 0) * 100)

  const runs = [
    { model: 'BiLSTM-CRF', detail: `${(state.entities || []).length} entities`, tone: 'verified' },
    { model: 'LegalBERT', detail: `${legalbert} contradictions`, tone: legalbert ? 'conflict' : 'court' },
    { model: 'XGBoost', detail: `${xgb} anomalies`, tone: xgb ? 'gold' : 'court' },
    { model: 'R-GCN', detail: `${integrity} · ${confidence}%`, tone: integrity === 'Intact' ? 'verified' : 'gold' }
  ]

  return (
    <div className="mb-6 rounded-lg border border-line bg-white/60 px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="font-mono text-[10.5px] uppercase tracking-widest text-textsoft">Models run</span>
        {runs.map((run) => (
          <span key={run.model} className="flex items-center gap-2 text-xs">
            <Badge tone={run.tone}>{run.model}</Badge>
            <span className="text-textsoft">{run.detail}</span>
          </span>
        ))}
        {structural > 0 && (
          <span className="flex items-center gap-2 text-xs">
            <Badge tone="gold">Entity consistency</Badge>
            <span className="text-textsoft">{structural} cross-document checks</span>
          </span>
        )}
      </div>
    </div>
  )
}

// --- Module 3: contradictions (LegalBERT) ----------------------------------

function ConflictList({ caseId }) {
  const [items, setItems] = useState([])
  useEffect(() => { getConflictMatrix(caseId).then(setItems).catch(() => {}) }, [caseId])

  return (
    <div className="space-y-4">
      {items.map((c) => (
        <Card key={c.id} className="p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="font-serif font-semibold text-ink">{c.type}</span>
              <Badge tone={SEVERITY_TONE[c.severity] || 'gold'}>{c.severity}</Badge>
            </div>
            <span className="text-[11px] font-mono text-textsoft">{c.model} · {Math.round(c.confidence * 100)}%</span>
          </div>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div className="bg-paperdim/40 rounded-md p-3">
              <div className="font-mono text-[11px] text-court mb-1">{c.left.source}</div>
              <div className="text-sm text-text">{c.left.claim}</div>
            </div>
            <div className="bg-paperdim/40 rounded-md p-3">
              <div className="font-mono text-[11px] text-conflict mb-1">{c.right.source}</div>
              <div className="text-sm text-text">{c.right.claim}</div>
            </div>
          </div>
          <p className="text-xs text-textsoft border-l-2 border-gold pl-3">{c.explanation}</p>
        </Card>
      ))}
      {items.length === 0 && <Card className="p-6 text-sm text-textsoft">No conflicts detected.</Card>}
    </div>
  )
}

// --- Module 3: anomalies (XGBoost) -----------------------------------------

function AnomalyList({ caseId }) {
  const [items, setItems] = useState([])
  useEffect(() => { getAnomalies(caseId).then(setItems).catch(() => {}) }, [caseId])

  return (
    <Card className="p-6 max-w-2xl">
      <div className="space-y-5">
        {items.map((a) => (
          <div key={a.id} className="border-b border-line pb-4 last:border-none">
            <div className="flex justify-between items-baseline gap-3 mb-1">
              <span className="text-sm font-medium text-ink">{a.entity}</span>
              <Badge tone={ANOMALY_TONE[a.type] || 'court'}>{a.type.replace(/_/g, ' ')}</Badge>
            </div>
            <div className="flex items-center gap-3 mb-2">
              <div className="h-1.5 bg-paperdim rounded-full overflow-hidden flex-1">
                <div className={`h-full ${a.score >= 0.75 ? 'bg-verified' : a.score >= 0.5 ? 'bg-gold' : 'bg-conflict'}`} style={{ width: `${Math.round(a.score * 100)}%` }} />
              </div>
              <span className="text-xs font-mono text-textsoft w-24 text-right">{a.model} {Math.round(a.score * 100)}%</span>
            </div>
            <p className="text-xs text-textsoft border-l-2 border-gold pl-3 mb-1">{a.explanation}</p>
            {a.sources?.length > 0 && (
              <div className="text-[11px] font-mono text-textsoft">Sources: {a.sources.join(' · ')}</div>
            )}
          </div>
        ))}
        {items.length === 0 && <p className="text-sm text-textsoft">No anomalies detected.</p>}
      </div>
    </Card>
  )
}

// --- Module 1-2: split-screen document viewer + anomaly overlay -------------

function Documents({ caseId }) {
  const [evidence, setEvidence] = useState([])
  const [selected, setSelected] = useState(null)
  const [doc, setDoc] = useState(null)

  useEffect(() => {
    getCase(caseId).then((c) => {
      const ev = c.evidenceDetail || []
      setEvidence(ev)
      if (ev.length) setSelected(ev[0].id)
    })
  }, [caseId])

  useEffect(() => {
    if (!selected) return
    setDoc(null)
    getDocumentAnalysis(selected).then(setDoc).catch(() => {})
  }, [selected])

  if (!selected) return <Card className="p-6 text-sm text-textsoft">No evidence uploaded yet.</Card>

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        {evidence.map((ev) => (
          <button
            key={ev.id}
            onClick={() => setSelected(ev.id)}
            className={`text-xs font-mono px-3 py-1.5 rounded-full border ${
              selected === ev.id ? 'bg-ink text-paper border-ink' : 'border-line text-textsoft hover:text-ink'
            }`}
          >
            {ev.id} · {ev.kind}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-5">
        <Card className="p-5 relative">
          <div className="text-[11px] font-mono uppercase text-textsoft mb-3">
            {doc?.kind === 'image' ? 'Scan — anomalies highlighted' : doc?.kind === 'audio' ? 'Audio transcript' : 'Document text'}
          </div>
          {doc?.kind === 'image' ? (
            <div className="relative bg-paperdim rounded-md aspect-[3/4] flex items-center justify-center text-xs text-textsoft">
              Document preview
              {(doc.anomalies || []).map((a, i) => (
                <div
                  key={i}
                  title={`${a.note} (${Math.round((a.confidence || 0) * 100)}%)`}
                  className="absolute border-2 border-conflict rounded-sm bg-conflict/10 cursor-help"
                  style={{ left: `${a.x}%`, top: `${a.y}%`, width: `${a.w}%`, height: `${a.h}%` }}
                />
              ))}
            </div>
          ) : (
            <div className="bg-paperdim/40 rounded-md p-4 text-sm leading-7 text-text max-h-[460px] overflow-y-auto">
              {doc?.transcript || doc?.rawText || '—'}
            </div>
          )}
          {doc?.kind === 'image' && <p className="text-[11px] text-textsoft mt-2 font-mono">Hover boxes for anomaly notes</p>}
        </Card>

        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase text-textsoft mb-3">BiLSTM-CRF entity extraction</div>
          {doc ? <AnnotatedText doc={doc} /> : <p className="text-xs text-textsoft">Loading…</p>}
        </Card>
      </div>
    </div>
  )
}

function AnnotatedText({ doc }) {
  const { rawText = '', entities = [] } = doc
  const sorted = [...entities].sort((a, b) => a.start - b.start)
  const parts = []
  let cursor = 0
  sorted.forEach((ent, i) => {
    if (ent.start > cursor) parts.push(<span key={`t-${i}`}>{rawText.slice(cursor, ent.start)}</span>)
    const style = ENTITY_STYLES[ent.type] || 'bg-paperdim text-textsoft'
    parts.push(
      <span key={`e-${i}`} className={`tag-highlight ${style}`}>
        {rawText.slice(ent.start, ent.end)}
        <span className="lbl">{ent.type}</span>
      </span>
    )
    cursor = ent.end
  })
  parts.push(<span key="last">{rawText.slice(cursor)}</span>)
  return <div className="text-sm leading-8 text-text">{parts}</div>
}

// --- Module 3: relationship graph (R-GCN + entity links) --------------------

const ENTITY_TYPE_COLORS = {
  PETITIONER: '#3B5B8C', RESPONDENT: '#A8384A', DEFENDANT: '#A8384A',
  WITNESS: '#3E6B54', JUDGE: '#8A6A20', LAWYER: '#B8863A',
  PERSON: '#6B4E9E', ORG: '#6B4E9E',
  COURT: '#3B5B8C', STATUTE: '#3E6B54', PROVISION: '#3E6B54',
  PRECEDENT: '#3E6B54', CASE_NUMBER: '#5B6478',
  DATE: '#8C5A9E', PLACE: '#8C5A9E', EVIDENCE: '#B8863A', OTHER: '#5B6478'
}

const GRAPH_LEGEND_FALLBACK = [
  { kind: 'document', label: 'Documents', tone: '#3B5B8C' },
  { kind: 'party', label: 'Parties & people', tone: '#B98A2E' },
  { kind: 'authority', label: 'Court & law', tone: '#3F7D6B' },
  { kind: 'fact', label: 'Facts & evidence', tone: '#8C5A9E' }
]

const HIDDEN_HANDLE = { width: 7, height: 7, background: '#C9CFDA', border: '1px solid #fff' }

function DocumentNode({ data, selected }) {
  const tone = data.flagged ? '#A8384A' : '#3B5B8C'
  return (
    <div
      title={`${data.label} — ${data.entityCount} linked entities`}
      className={`w-[228px] rounded-xl border-2 bg-white shadow-[0_1px_3px_rgba(16,25,42,0.08)] ${selected ? 'ring-2 ring-gold ring-offset-1' : ''}`}
      style={{ borderColor: tone }}
    >
      <Handle id="tl" type="target" position={Position.Left} style={HIDDEN_HANDLE} />
      <Handle id="tr" type="target" position={Position.Right} style={HIDDEN_HANDLE} />
      <Handle id="sl" type="source" position={Position.Left} style={HIDDEN_HANDLE} />
      <Handle id="sr" type="source" position={Position.Right} style={HIDDEN_HANDLE} />
      <div className="flex items-center gap-2 px-3 pt-2.5">
        <span className="text-[15px] leading-none">{data.docType === 'image' ? '🖼️' : '📄'}</span>
        <span className="min-w-0 flex-1 truncate font-sans text-[12.5px] font-semibold text-text">{data.label}</span>
        {data.flagged && (
          <span className="rounded-full bg-conflictsoft px-1.5 py-[1px] font-mono text-[9px] font-semibold uppercase text-conflict">risk</span>
        )}
      </div>
      <div className="px-3 pb-2.5 pt-1">
        <div className="font-mono text-[9.5px] uppercase tracking-wide" style={{ color: tone }}>
          {data.entityCount} linked {data.entityCount === 1 ? 'entity' : 'entities'}
        </div>
        {data.excerpt && <div className="mt-1 text-[10.5px] leading-4 text-textsoft">{data.excerpt}</div>}
      </div>
    </div>
  )
}

function EntityNode({ data, selected }) {
  const color = ENTITY_TYPE_COLORS[data.entityType] || '#5B6478'
  return (
    <div
      title={`${data.label} · ${data.entityType} · ${data.mentions} mention(s) across ${data.docCount} document(s)`}
      className={`w-[196px] rounded-lg border bg-white shadow-[0_1px_3px_rgba(16,25,42,0.06)] ${selected ? 'ring-2 ring-gold ring-offset-1' : ''}`}
      style={{ borderColor: '#E7E1D0' }}
    >
      <Handle id="l" type="target" position={Position.Left} style={HIDDEN_HANDLE} />
      <Handle id="r" type="source" position={Position.Right} style={HIDDEN_HANDLE} />
      <div className="flex items-center gap-2 px-2.5 pt-2">
        <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ background: color }} />
        <span className="min-w-0 flex-1 truncate text-[12px] font-medium text-text">{data.label}</span>
      </div>
      <div className="flex items-center justify-between gap-2 px-2.5 pb-2 pt-1">
        <span className="rounded px-1.5 py-[1px] font-mono text-[9px] font-semibold uppercase" style={{ background: `${color}1A`, color }}>
          {data.entityType}
        </span>
        <span className="font-mono text-[9.5px] text-textsoft">
          {data.docCount} doc{data.docCount === 1 ? '' : 's'} · {data.mentions}×
        </span>
      </div>
    </div>
  )
}

function MentionEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, data }) {
  const [path] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  const dim = data?.dim
  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={dim ? undefined : markerEnd}
      style={{ stroke: '#B4BCCB', strokeWidth: 1.3, opacity: dim ? 0.1 : 0.55 }}
    />
  )
}

function ConflictEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd }) {
  const [path] = getSmoothStepPath({
    sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, borderRadius: 18
  })
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{ stroke: '#A8384A', strokeWidth: 2.2, strokeDasharray: '7 5' }}
      />
    </>
  )
}

const graphNodeTypes = { documentNode: DocumentNode, entityNode: EntityNode }
const graphEdgeTypes = { mentionEdge: MentionEdge, conflictEdge: ConflictEdge }

function GraphPlaceholder({ text }) {
  return (
    <div className="flex h-full items-center justify-center">
      <span className="font-mono text-[11.5px] text-textsoft">{text}</span>
    </div>
  )
}

function KnowledgeGraph({ caseId }) {
  const [graph, setGraph] = useState({ nodes: [], edges: [], meta: {}, legend: [] })
  const [selected, setSelected] = useState(null)
  const [showMentions, setShowMentions] = useState(true)
  const [status, setStatus] = useState('loading')

  useEffect(() => {
    setStatus('loading')
    setSelected(null)
    getGraph(caseId)
      .then((g) => {
        setGraph({
          nodes: g?.nodes || [], edges: g?.edges || [],
          meta: g?.meta || {}, legend: g?.legend || []
        })
        setStatus('ready')
      })
      .catch(() => setStatus('error'))
  }, [caseId])

  const neighbors = useMemo(() => {
    const map = new Map()
    for (const edge of graph.edges) {
      if (!map.has(edge.source)) map.set(edge.source, new Set())
      if (!map.has(edge.target)) map.set(edge.target, new Set())
      map.get(edge.source).add(edge.target)
      map.get(edge.target).add(edge.source)
    }
    return map
  }, [graph.edges])

  const conflictNodes = useMemo(
    () => new Set(graph.edges.filter((e) => e.type === 'conflictEdge').flatMap((e) => [e.source, e.target])),
    [graph.edges]
  )

  const conflictLabels = useMemo(
    () => [...new Set(graph.edges
      .filter((e) => e.type === 'conflictEdge')
      .map((e) => e.data?.label)
      .filter(Boolean))],
    [graph.edges]
  )

  const nodes = useMemo(() => {
    const focus = selected ? new Set([selected, ...(neighbors.get(selected) || [])]) : null
    return graph.nodes.map((node) => ({
      ...node,
      draggable: false,
      selected: node.id === selected,
      style: {
        ...(node.style || {}),
        opacity: focus && !focus.has(node.id) ? 0.2 : 1,
        transition: 'opacity 180ms ease'
      }
    }))
  }, [graph.nodes, neighbors, selected])

  const edges = useMemo(() => {
    const focus = selected ? new Set([selected, ...(neighbors.get(selected) || [])]) : null
    return graph.edges
      .filter((edge) => showMentions || edge.type !== 'mentionEdge')
      .map((edge) => {
        const conflict = edge.type === 'conflictEdge'
        const dim = focus ? !(focus.has(edge.source) && focus.has(edge.target)) : false
        return {
          ...edge,
          sourceHandle: conflict ? 'sl' : 'sr',
          targetHandle: conflict ? 'tl' : 'l',
          data: { ...(edge.data || {}), dim },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 15,
            height: 15,
            color: conflict ? '#A8384A' : '#9AA3B4'
          }
        }
      })
  }, [graph.edges, neighbors, selected, showMentions])

  const meta = graph.meta || {}
  const legend = graph.legend?.length ? graph.legend : GRAPH_LEGEND_FALLBACK
  const suspicious = meta.integrity === 'Suspicious'
  const integrityTone = suspicious
    ? 'bg-conflictsoft text-conflict'
    : meta.integrity === 'Intact'
      ? 'bg-verifiedsoft text-verified'
      : 'bg-paperdim text-textsoft'

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-paperdim/40 px-5 py-3">
        <span className="font-mono text-[11px] uppercase tracking-widest text-gold">Evidence knowledge graph</span>
        <span className="font-mono text-[11px] text-textsoft">
          {meta.documents ?? 0} documents · {meta.entities ?? 0} entities · {meta.mentions ?? 0} links · {meta.conflicts ?? 0} conflicts
        </span>
        <span className={`rounded-full px-2 py-0.5 font-mono text-[10.5px] font-semibold ${integrityTone}`}>
          {meta.model || 'Graph'} · {meta.integrity || 'unscored'}
        </span>
        <label className="ml-auto flex cursor-pointer select-none items-center gap-2 font-mono text-[10.5px] uppercase tracking-wide text-textsoft">
          <input
            type="checkbox"
            checked={showMentions}
            onChange={(e) => setShowMentions(e.target.checked)}
            className="accent-[#B8863A]"
          />
          Entity links
        </label>
      </div>

      <div className="relative h-[600px] bg-[#FBF9F4]">
        {status === 'loading' && <GraphPlaceholder text="Loading graph…" />}
        {status === 'error' && <GraphPlaceholder text="Could not load the graph for this case." />}
        {status === 'ready' && nodes.length === 0 && <GraphPlaceholder text="Run the analysis to build the entity graph." />}
        {status === 'ready' && nodes.length > 0 && (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={graphNodeTypes}
            edgeTypes={graphEdgeTypes}
            onNodeClick={(_, node) => setSelected((cur) => (cur === node.id ? null : node.id))}
            onPaneClick={() => setSelected(null)}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            onInit={(instance) => setTimeout(() => instance.fitView({ padding: 0.2 }), 60)}
            zoomOnScroll={false}
            preventScrolling={false}
            minZoom={0.25}
            maxZoom={1.6}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#E4DECC" gap={22} size={1} />
            <Controls showInteractive={false} />
            <MiniMap
              pannable
              zoomable
              maskColor="rgba(246,243,236,0.75)"
              nodeColor={(n) => (n.type === 'documentNode'
                ? '#3B5B8C'
                : ENTITY_TYPE_COLORS[n.data?.entityType] || '#5B6478')}
            />
            <Panel position="top-left">
              <div className="flex flex-wrap gap-x-3 gap-y-1 rounded-lg border border-line bg-white/90 px-3 py-2 shadow-sm backdrop-blur">
                {legend.map((item) => (
                  <span key={item.kind} className="flex items-center gap-1.5 font-mono text-[9.5px] uppercase tracking-wide text-textsoft">
                    <span className="h-2 w-2 rounded-full" style={{ background: item.tone }} />
                    {item.label}
                  </span>
                ))}
                <span className="flex items-center gap-1.5 font-mono text-[9.5px] uppercase tracking-wide text-conflict">
                  <span className="h-[2px] w-4 bg-conflict" /> conflict link
                </span>
              </div>
            </Panel>
          </ReactFlow>
        )}
      </div>

      {conflictNodes.size > 0 && (
        <div className="border-t border-line px-5 py-2 font-mono text-[10.5px] text-textsoft">
          {conflictNodes.size} exhibits linked by a conflict
          {conflictLabels.length > 0 ? ` (${conflictLabels.join(', ')})` : ''} · click a node to trace its connections
        </div>
      )}
    </Card>
  )
}

// --- Module 2: entity inventory (BiLSTM-CRF output) ------------------------

function Entities({ caseId }) {
  const [items, setItems] = useState([])
  useEffect(() => { getEntities(caseId).then(setItems).catch(() => {}) }, [caseId])

  const groups = items.reduce((acc, e) => {
    acc[e.type] = acc[e.type] || []
    acc[e.type].push(e)
    return acc
  }, {})

  return (
    <Card className="p-6">
      <div className="mb-5 flex flex-wrap gap-2">
        {Object.entries(groups).map(([type, list]) => (
          <span key={type} className={`text-[11px] font-mono px-2.5 py-1 rounded-full ${ENTITY_STYLES[type] || 'bg-paperdim text-textsoft'}`}>
            {type} · {list.length}
          </span>
        ))}
      </div>
      <table className="w-full">
        <thead>
          <tr className="border-b border-line">
            {['Entity', 'Type', 'Confidence'].map((h) => (
              <th key={h} className="text-left text-[11px] uppercase tracking-wide text-textsoft font-semibold px-4 py-3">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((e, i) => (
            <tr key={i} className="border-b border-line last:border-none">
              <td className="px-4 py-3 text-sm text-ink">{e.text}</td>
              <td className="px-4 py-3">
                <span className={`text-[10.5px] font-mono font-semibold px-2 py-1 rounded-full ${ENTITY_STYLES[e.type] || 'bg-paperdim text-textsoft'}`}>{e.type}</span>
              </td>
              <td className="px-4 py-3 text-sm font-mono text-textsoft">{Math.round((e.confidence || 0) * 100)}%</td>
            </tr>
          ))}
          {items.length === 0 && (
            <tr><td colSpan={3} className="px-4 py-6 text-center text-sm text-textsoft">No entities yet — run the pipeline first.</td></tr>
          )}
        </tbody>
      </table>
    </Card>
  )
}

// --- Module 4: explainable report ------------------------------------------

function Report({ caseId }) {
  const [report, setReport] = useState(null)
  useEffect(() => { getReportContent(caseId).then(setReport).catch(() => {}) }, [caseId])

  function downloadJSON() {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${(report?.case?.caseNumber || caseId).replace(/[^A-Za-z0-9]+/g, '_')}-conflict-report.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!report) return <Card className="p-6 text-sm text-textsoft">Generating report…</Card>

  const s = report.stats || {}
  return (
    <div>
      <div className="flex gap-3 mb-5 no-print">
        <Button onClick={() => window.print()}>Print / Save as PDF</Button>
        <Button variant="ghost" onClick={downloadJSON}>Download JSON</Button>
      </div>

      <Card className="p-8 max-w-3xl">
        <div className="border-b border-line pb-4 mb-5">
          <div className="font-mono text-[11px] uppercase tracking-widest text-gold mb-1">Explainable Conflict-Resolution Report</div>
          <h2 className="font-serif text-2xl text-ink">{report.case.caseNumber || caseId}</h2>
          <p className="text-textsoft text-sm">{report.case.court} · IPC/BNS {report.case.ipcSections}</p>
          <p className="text-[11px] font-mono text-textsoft mt-1">
            Generated {new Date(report.generatedAt).toLocaleString()}
          </p>
        </div>

        <p className="text-sm text-text mb-5 leading-7">{report.summary}</p>

        <div className="grid grid-cols-4 gap-3 mb-6">
          <Metric label="Documents" value={s.documents} />
          <Metric label="Entities" value={s.entities} />
          <Metric label="Contradictions" value={s.conflicts} />
          <Metric label="Anomalies" value={s.anomalies} />
        </div>

        <h3 className="font-serif font-semibold text-ink mb-2">Detected conflicts</h3>
        <ul className="text-sm text-text leading-7 list-disc pl-5 mb-6">
          {report.conflicts.map((c) => (
            <li key={c.id}>
              <b>{c.type}:</b> {c.left.claim} → {c.right.claim}{' '}
              <span className="text-textsoft">({c.model})</span>
            </li>
          ))}
          {report.conflicts.length === 0 && <li className="list-none text-textsoft">None.</li>}
        </ul>

        <h3 className="font-serif font-semibold text-ink mb-2">Anomalies &amp; explanations</h3>
        <ul className="text-sm text-text leading-7 list-disc pl-5 mb-6">
          {report.anomalies.map((a) => (
            <li key={a.id}>
              <b>{a.type.replace(/_/g, ' ')}</b> — {a.entity}: {a.explanation}{' '}
              <span className="text-textsoft">({a.model} {Math.round(a.score * 100)}%)</span>
            </li>
          ))}
          {report.anomalies.length === 0 && <li className="list-none text-textsoft">None.</li>}
        </ul>

        <h3 className="font-serif font-semibold text-ink mb-2">Recommended next steps</h3>
        <ol className="text-sm text-text leading-7 list-decimal pl-5">
          {report.recommendations.map((r, i) => <li key={i}>{r}</li>)}
        </ol>

        <p className="text-[11px] text-textsoft mt-6 pt-4 border-t border-line">
          This report is decision support only. All findings require human verification before use in proceedings.
        </p>
      </Card>
    </div>
  )
}
