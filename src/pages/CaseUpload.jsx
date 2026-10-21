import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { getCase, runAnalysis, tagEvidence, uploadEvidence } from '../lib/api'
import { Badge, Button, Card, Eyebrow } from '../components/ui/Primitives'

export default function CaseUpload() {
  const { caseId } = useParams()
  const [caseInfo, setCaseInfo] = useState(null)
  const [files, setFiles] = useState([])
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [evidence, setEvidence] = useState([])
  const [analyzing, setAnalyzing] = useState(false)
  const [stage, setStage] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    getCase(caseId).then((c) => {
      setCaseInfo(c)
      setEvidence(c.evidenceDetail || [])
    })
  }, [caseId])

  const people = caseInfo?.people || []

  async function handleFiles(fileList) {
    const list = Array.from(fileList)
    if (list.length === 0) return
    setUploading(true)
    setProgress(0)
    try {
      const uploaded = await uploadEvidence(caseId, list, setProgress)
      setEvidence((e) => [...e, ...uploaded])
    } finally {
      setUploading(false)
      setFiles([])
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragActive(false)
    handleFiles(e.dataTransfer.files)
  }

  async function linkPerson(evidenceId, personId) {
    await tagEvidence(evidenceId, personId)
    setEvidence((list) => list.map((ev) => (ev.id === evidenceId ? { ...ev, linkedTo: personId } : ev)))
  }

  async function handleAnalyze() {
    setAnalyzing(true)
    await runAnalysis(caseId, setStage)
    setAnalyzing(false)
    navigate(`/cases/${caseId}/analysis`)
  }

  return (
    <div className="max-w-2xl">
      <Eyebrow>Module 1 — Ingestion · {caseInfo?.caseNumber || caseId}</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">Upload Evidence</h1>
      <p className="text-textsoft text-sm mb-7">
        PDFs, images, and audio transcripts all go through the same pipeline before BIO-tagging.
      </p>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-xl px-6 py-14 text-center transition-colors ${
          dragActive ? 'border-gold bg-goldsoft/40' : 'border-line bg-white'
        }`}
      >
        <div className="font-mono text-xs text-gold tracking-wide mb-2">.PDF · .PNG/.JPG · .MP3/.WAV</div>
        <h3 className="font-serif font-semibold text-ink mb-1">Drop files or click to browse</h3>
        <p className="text-textsoft text-xs mb-4">Audio transcripts are auto-transcribed before entity extraction</p>
        <input id="file-input" type="file" multiple className="hidden" onChange={(e) => handleFiles(e.target.files)} />
        <Button variant="ghost" type="button" onClick={() => document.getElementById('file-input').click()}>
          Select files
        </Button>
      </div>

      {uploading && (
        <div className="mt-4">
          <div className="h-1.5 w-full bg-paperdim rounded-full overflow-hidden">
            <div className="h-full bg-gold transition-all" style={{ width: `${progress}%` }} />
          </div>
          <p className="text-xs text-textsoft mt-1.5 font-mono">Uploading… {progress}%</p>
        </div>
      )}

      {evidence.length > 0 && (
        <Card className="mt-6 p-5">
          <h3 className="font-serif font-semibold text-ink mb-3">Evidence ({evidence.length}) — link each item to a person</h3>
          {evidence.map((ev) => (
            <div key={ev.id} className="flex items-center justify-between gap-3 py-2.5 border-b border-line last:border-none">
              <div className="text-sm flex items-center gap-2">
                <span className="font-mono text-[11px] text-gold">{ev.id}</span>
                <span>{ev.name}</span>
                {ev.kind && <Badge tone="court">{ev.kind}</Badge>}
              </div>
              <select
                className="border border-line rounded-md px-2.5 py-1.5 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-gold"
                value={ev.linkedTo || ''}
                onChange={(e) => linkPerson(ev.id, e.target.value)}
              >
                <option value="">Link to person…</option>
                {people.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.role})</option>
                ))}
              </select>
            </div>
          ))}

          <Button className="mt-5 w-full" onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? 'Analyzing…' : 'Run AI analysis'}
          </Button>
        </Card>
      )}

      {analyzing && (
        <Card className="mt-4 p-5 flex items-center gap-4">
          <div className="relative w-8 h-8 shrink-0">
            <div className="absolute inset-0 border-2 border-paperdim rounded-full" />
            <div className="absolute inset-0 border-2 border-gold border-t-transparent rounded-full animate-spin" />
          </div>
          <div>
            <div className="text-sm font-semibold text-ink">{stage || 'Starting pipeline…'}</div>
            <div className="text-xs text-textsoft font-mono">Multi-model reasoning engine active</div>
          </div>
        </Card>
      )}

      <p className="text-xs text-textsoft mt-6">
        <Link to={`/cases/${caseId}`} className="text-gold font-semibold">← Back to case</Link>
      </p>
    </div>
  )
}
