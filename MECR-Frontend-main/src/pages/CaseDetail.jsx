import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { deleteEvidence, getCase, runAnalysis, tagEvidence, checkForgery } from '../lib/api'
import { Badge, Button, Card, Eyebrow } from '../components/ui/Primitives'

export default function CaseDetail() {
  const { caseId } = useParams()
  const [caseInfo, setCaseInfo] = useState(null)
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const [forgeryResults, setForgeryResults] = useState({})
  const [forgeryLoading, setForgeryLoading] = useState({})
  const navigate = useNavigate()

  async function load() {
    try {
      setCaseInfo(await getCase(caseId))
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => { load() }, [caseId])

  const people = caseInfo?.people || []
  const evidence = caseInfo?.evidenceDetail || []

  async function linkPerson(evidenceId, personId) {
    await tagEvidence(evidenceId, personId)
    load()
  }

  async function removeEvidence(id) {
    await deleteEvidence(id)
    load()
  }

  async function handleScreen(evidenceId) {
    setForgeryLoading((p) => ({ ...p, [evidenceId]: true }))
    try {
      const data = await checkForgery(evidenceId)
      setForgeryResults((p) => ({ ...p, [evidenceId]: data }))
    } catch (e) {
      setForgeryResults((p) => ({ ...p, [evidenceId]: { error: e.message } }))
    } finally {
      setForgeryLoading((p) => ({ ...p, [evidenceId]: false }))
    }
  }

  async function handleAnalyze() {
    setRunning(true)
    setStage('')
    try {
      await runAnalysis(caseId, setStage)
      await load()
      navigate(`/cases/${caseId}/analysis`)
    } finally {
      setRunning(false)
    }
  }

  if (error) return <p className="text-conflict text-sm">{error}</p>
  if (!caseInfo) return <p className="text-textsoft text-sm">Loading case…</p>

  return (
    <div>
      <Eyebrow>Case File</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">{caseInfo.caseNumber || caseInfo.id}</h1>
      <p className="text-textsoft text-sm mb-6 flex items-center gap-2">
        {caseInfo.court} · IPC/BNS {caseInfo.ipcSections || '—'}
        <Badge tone={caseInfo.status === 'active' ? 'court' : 'gold'}>{caseInfo.status}</Badge>
      </p>

      <div className="flex gap-3 mb-7">
        <Button onClick={() => navigate(`/cases/${caseId}/upload`)}>Upload evidence</Button>
        <Button variant="ghost" onClick={handleAnalyze} disabled={running || evidence.length === 0}>
          {running ? 'Analyzing…' : 'Run AI analysis'}
        </Button>
        <Button variant="ghost" onClick={() => navigate(`/cases/${caseId}/analysis`)} disabled={!caseInfo.analysed}>
          View analysis
        </Button>
      </div>

      {running && (
        <Card className="mb-6 p-4 text-sm text-textsoft">
          <span className="font-mono text-[11px] uppercase text-gold mr-2">Pipeline</span>
          {stage || 'starting…'}
        </Card>
      )}

      <div className="grid grid-cols-2 gap-5 mb-6">
        <Card className="p-5">
          <h3 className="font-serif font-semibold text-ink mb-3">People ({people.length})</h3>
          {people.map((p) => (
            <div key={p.id} className="flex justify-between items-center py-2 border-b border-line last:border-none text-sm">
              <span>{p.name}</span>
              <Badge tone={p.role === 'Defendant' ? 'conflict' : 'verified'}>{p.role}</Badge>
            </div>
          ))}
          {people.length === 0 && <p className="text-xs text-textsoft">No people recorded.</p>}
        </Card>

        <Card className="p-5">
          <h3 className="font-serif font-semibold text-ink mb-3">Summary</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div><div className="text-textsoft text-xs">Evidence</div><div className="font-mono text-lg">{evidence.length}</div></div>
            <div><div className="text-textsoft text-xs">Conflicts</div><div className="font-mono text-lg text-conflict">{caseInfo.conflicts}</div></div>
            <div><div className="text-textsoft text-xs">Anomalies</div><div className="font-mono text-lg text-gold">{caseInfo.anomalies}</div></div>
            <div><div className="text-textsoft text-xs">Analysed</div><div className="font-mono text-lg">{caseInfo.analysed ? 'Yes' : 'No'}</div></div>
          </div>
        </Card>
      </div>

      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Evidence ({evidence.length})</h3>
        <table className="w-full">
          <thead>
            <tr className="border-b border-line">
              {['ID', 'Name', 'Kind', 'Linked to', 'Forgery Screen (exp.)', ''].map((h) => (
                <th key={h} className="text-left text-[11px] uppercase tracking-wide text-textsoft font-semibold px-3 py-2.5">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {evidence.map((ev) => (
              <tr key={ev.id} className="border-b border-line last:border-none">
                <td className="px-3 py-2.5 font-mono text-[11px] text-gold">{ev.id}</td>
                <td className="px-3 py-2.5 text-sm text-ink">{ev.name}</td>
                <td className="px-3 py-2.5"><Badge tone="court">{ev.kind}</Badge></td>
                <td className="px-3 py-2.5">
                  <select
                    className="border border-line rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-gold"
                    value={ev.linkedTo || ''}
                    onChange={(e) => linkPerson(ev.id, e.target.value)}
                  >
                    <option value="">Unlinked</option>
                    {people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                </td>
                <td className="px-3 py-2.5">
                  {ev.kind === 'image' ? (
                    <div className="min-w-[130px]">
                      {!forgeryResults[ev.id] && (
                        <button
                          id={`forgery-screen-${ev.id}`}
                          onClick={() => handleScreen(ev.id)}
                          disabled={!!forgeryLoading[ev.id]}
                          className="text-[11px] font-mono px-2.5 py-1 rounded border border-line text-textsoft hover:text-ink hover:border-gold transition-colors disabled:opacity-50"
                        >
                          {forgeryLoading[ev.id] ? 'Screening…' : '⬡ Screen'}
                        </button>
                      )}
                      {forgeryResults[ev.id] && (() => {
                        const r = forgeryResults[ev.id]
                        if (r.error) return <span className="text-[11px] font-mono text-conflict" title={r.error}>Error</span>
                        const tone = r.raw_result === 'LIKELY AUTHENTIC' ? 'verified'
                          : r.raw_result === 'POTENTIALLY TAMPERED' ? 'conflict' : 'gold'
                        return (
                          <div className="flex items-center gap-1.5" title={r.disclaimer || 'Experimental screening only — not proof of authenticity.'}>
                            <Badge tone={tone}>{r.result}</Badge>
                            <button
                              onClick={() => handleScreen(ev.id)}
                              disabled={!!forgeryLoading[ev.id]}
                              title="Re-screen"
                              className="text-[10px] text-textsoft hover:text-ink"
                            >↻</button>
                          </div>
                        )
                      })()}
                    </div>
                  ) : (
                    <span className="text-[11px] font-mono text-textsoft/50">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-right">
                  <button onClick={() => removeEvidence(ev.id)} className="text-xs text-textsoft hover:text-conflict">Delete</button>
                </td>
              </tr>
            ))}
            {evidence.length === 0 && (
              <tr><td colSpan={6} className="px-3 py-6 text-center text-sm text-textsoft">
                No evidence yet. <Link to={`/cases/${caseId}/upload`} className="text-gold font-semibold">Upload some</Link>.
              </td></tr>
            )}
          </tbody>
        </table>
        <p className="mt-3 px-1 text-[11px] text-textsoft border-l-2 border-gold pl-3">
          Forgery screening is an experimental, secondary signal. It is not proof of
          authenticity or tampering and must be verified against original records.
        </p>
      </Card>
    </div>
  )
}
