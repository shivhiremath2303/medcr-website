import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createCase } from '../lib/api'
import { Button, Card, Eyebrow, Input } from '../components/ui/Primitives'

export default function CaseCreate() {
  const [caseId, setCaseId] = useState('')
  const [court, setCourt] = useState('')
  const [ipcSections, setIpcSections] = useState('')
  const [people, setPeople] = useState([{ name: '', role: 'Defendant' }])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  function updatePerson(i, field, value) {
    setPeople((p) => p.map((person, idx) => (idx === i ? { ...person, [field]: value } : person)))
  }

  function addPerson() {
    setPeople((p) => [...p, { name: '', role: 'Defendant' }])
  }

  function removePerson(i) {
    setPeople((p) => (p.length > 1 ? p.filter((_, idx) => idx !== i) : p))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const data = await createCase({
        caseId,
        court,
        ipcSections,
        defendants: people.filter((p) => p.name.trim()),
      })
      navigate(`/cases/${data.id}/upload`)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl">
      <Eyebrow>Module 1 — New Filing</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">Create Case</h1>
      <p className="text-textsoft text-sm mb-7">
        Register the case before uploading evidence — this becomes the anchor every document links back to.
      </p>

      <Card className="p-7">
        <form onSubmit={handleSubmit}>
          <Input label="Case Number" placeholder="Crl MC 1622/2006" value={caseId} onChange={(e) => setCaseId(e.target.value)} required />
          <Input label="Court" placeholder="High Court of Kerala" value={court} onChange={(e) => setCourt(e.target.value)} />
          <Input label="IPC / BNS Sections" placeholder="e.g. 302, 34, 120B" value={ipcSections} onChange={(e) => setIpcSections(e.target.value)} required />

          <div className="mb-2">
            <span className="block text-xs font-mono uppercase tracking-wide text-textsoft mb-1.5">Defendants / Witnesses</span>
            {people.map((person, i) => (
              <div key={i} className="flex gap-2 mb-2">
                <input
                  className="flex-1 border border-line rounded-md px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-gold"
                  placeholder="Full name"
                  value={person.name}
                  onChange={(e) => updatePerson(i, 'name', e.target.value)}
                />
                <select
                  className="border border-line rounded-md px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-gold"
                  value={person.role}
                  onChange={(e) => updatePerson(i, 'role', e.target.value)}
                >
                  <option>Defendant</option>
                  <option>Witness</option>
                </select>
                {people.length > 1 && (
                  <button type="button" onClick={() => removePerson(i)} className="text-textsoft hover:text-conflict px-1" title="Remove">
                    ✕
                  </button>
                )}
              </div>
            ))}
            <button type="button" onClick={addPerson} className="text-xs font-mono text-gold font-semibold mt-1">
              + Add another person
            </button>
          </div>

          {error && <p className="text-conflict text-xs mb-4">{error}</p>}
          <Button type="submit" className="w-full mt-5" disabled={loading}>
            {loading ? 'Creating…' : 'Create case & continue to upload'}
          </Button>
        </form>
      </Card>
    </div>
  )
}
