import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { listCases } from '../lib/api'
import { Badge, Card, Eyebrow } from '../components/ui/Primitives'

function Stat({ label, value, tone = 'text-ink' }) {
  return (
    <Card className="px-5 py-4">
      <div className="text-[11px] font-mono uppercase tracking-wide text-textsoft mb-1">{label}</div>
      <div className={`font-serif text-2xl font-semibold ${tone}`}>{value}</div>
    </Card>
  )
}

export default function Dashboard() {
  const [cases, setCases] = useState([])
  const [filter, setFilter] = useState('active')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    listCases()
      .then(setCases)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const filtered = cases.filter((c) => c.status === filter)
  const totals = cases.reduce(
    (acc, c) => ({
      evidence: acc.evidence + (c.evidence || 0),
      conflicts: acc.conflicts + (c.conflicts || 0),
      anomalies: acc.anomalies + (c.anomalies || 0),
    }),
    { evidence: 0, conflicts: 0, anomalies: 0 }
  )

  return (
    <div>
      <Eyebrow>Case Overview</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">Case Files</h1>
      <p className="text-textsoft text-sm mb-7 max-w-lg">
        Cases created by your account, with anomaly counts from the last analysis run.
        Each user's case files are private.
      </p>

      <div className="grid grid-cols-4 gap-4 mb-7">
        <Stat label="Cases" value={cases.length} />
        <Stat label="Evidence items" value={totals.evidence} />
        <Stat label="Conflicts" value={totals.conflicts} tone="text-conflict" />
        <Stat label="Anomalies" value={totals.anomalies} tone="text-gold" />
      </div>

      <div className="flex gap-2 mb-5">
        {['active', 'archived'].map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`text-xs font-mono uppercase tracking-wide px-3 py-1.5 rounded-full border ${
              filter === s ? 'bg-ink text-paper border-ink' : 'border-line text-textsoft'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      <Card>
        {error && (
          <p className="px-4 py-3 text-sm text-conflict border-b border-line">{error}</p>
        )}
        <table className="w-full">
          <thead>
            <tr className="border-b border-line">
              {['Case ID', 'Court', 'Defendants', 'Evidence', 'Findings', 'Status'].map((h) => (
                <th key={h} className="text-left text-[11px] uppercase tracking-wide text-textsoft font-semibold px-4 py-3">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((c) => (
              <tr
                key={c.id}
                onClick={(e) => {
                  if (e.target.closest('a')) return
                  navigate(`/cases/${c.id}`)
                }}
                className="border-b border-line last:border-none hover:bg-paperdim/40 cursor-pointer"
              >
                <td className="px-4 py-3.5 text-sm">
                  <Link to={`/cases/${c.id}`} className="font-medium text-ink hover:text-gold">
                    {c.caseNumber || c.id}
                  </Link>
                </td>
                <td className="px-4 py-3.5 text-sm text-textsoft">{c.court}</td>
                <td className="px-4 py-3.5 text-sm text-textsoft">{c.defendants}</td>
                <td className="px-4 py-3.5 text-sm text-textsoft font-mono">{c.evidence}</td>
                <td className="px-4 py-3.5 text-sm">
                  {c.conflicts > 0 || c.anomalies > 0 ? (
                    <span className="flex gap-1.5">
                      {c.conflicts > 0 && <Badge tone="conflict">{c.conflicts} conflicts</Badge>}
                      {c.anomalies > 0 && <Badge tone="gold">{c.anomalies} anomalies</Badge>}
                    </span>
                  ) : (
                    <Badge tone="verified">Clean</Badge>
                  )}
                </td>
                <td className="px-4 py-3.5">
                  <Badge tone={c.status === 'active' ? 'court' : 'gold'}>{c.status}</Badge>
                </td>
              </tr>
            ))}
            {!loading && filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-textsoft">
                  No {filter} cases yet. <Link to="/cases/new" className="text-gold font-semibold">Create one</Link>.
                </td>
              </tr>
            )}
            {loading && (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-sm text-textsoft">Loading cases…</td></tr>
            )}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
