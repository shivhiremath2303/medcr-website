import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  adminMe, adminListUsers, adminCreateUser, adminToggleRole, adminResetPassword, adminDeleteUser,
  adminUploadDataset, adminListDatasets, adminDownloadDataset,
  adminDatasetCatalog, adminFetchDataset,
  adminTrainModel, adminTrainingStatus,
  adminListModels, adminActivateModel,
  logout, isAdmin
} from '../lib/api'
import { Button, Input, Badge, Card, Eyebrow } from '../components/ui/Primitives'

const MODEL_TYPES = [
  { key: 'bilstm_crf', label: 'BiLSTM-CRF NER', desc: 'Legal entity extraction' },
  { key: 'xgboost', label: 'XGBoost Anomaly', desc: 'Anomaly detection' },
  { key: 'legalbert', label: 'LegalBERT', desc: 'Contradiction detection' },
  { key: 'rgcn', label: 'R-GCN', desc: 'Conflict knowledge graph' },
]

const TABS = [
  { key: 'dashboard', label: 'Overview' },
  { key: 'users', label: 'User Management' },
  { key: 'datasets', label: 'Datasets' },
  { key: 'training', label: 'Model Training' },
  { key: 'models', label: 'Deployed Models' },
]

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [me, setMe] = useState(null)
  const [users, setUsers] = useState([])
  const [datasets, setDatasets] = useState([])
  const [models, setModels] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  const [newUser, setNewUser] = useState({ name: '', email: '', password: '', isAdmin: false })
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadingDataset, setUploadingDataset] = useState(false)
  const [catalog, setCatalog] = useState([])
  const [fetchingDataset, setFetchingDataset] = useState('')
  const [downloadingDataset, setDownloadingDataset] = useState('')
  const [training, setTraining] = useState({})
  const [trainResult, setTrainResult] = useState(null)

  const navigate = useNavigate()

  async function load() {
    try {
      setLoading(true)
      const [m, list, ds, md, cat] = await Promise.all([
        adminMe(), adminListUsers(), adminListDatasets(), adminListModels(),
        adminDatasetCatalog().catch(() => [])
      ])
      setMe(m)
      setUsers(list)
      setDatasets(ds)
      setModels(md)
      setCatalog(cat)
      setInfo(`Signed in as ${m.name} (${m.email}) — Administrator`)
    } catch (err) {
      setError(err.message)
      navigate('/login')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!isAdmin()) {
      navigate('/login')
      return
    }
    load()
  }, [])

  function updateUser(field) {
    return (e) => setNewUser({ ...newUser, [field]: e.target.value })
  }

  async function handleCreateUser(e) {
    e.preventDefault()
    setError('')
    try {
      await adminCreateUser({
        name: newUser.name,
        email: newUser.email,
        password: newUser.password,
        isAdmin: newUser.isAdmin,
      })
      setNewUser({ name: '', email: '', password: '', isAdmin: false })
      await load()
      setInfo(`Created ${newUser.email}`)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleToggleRole(email, current) {
    setError('')
    try {
      await adminToggleRole(email, !current)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleResetPassword(email) {
    const password = window.prompt(
      `New password for ${email}\nMinimum 8 characters with upper case, lower case, a digit and a symbol.`
    )
    if (!password) return
    setError('')
    try {
      await adminResetPassword(email, password)
      if (me?.email === email) {
        setInfo('Password updated. Please sign in again.')
        logout()
        navigate('/login')
        return
      }
      setInfo(`Password updated for ${email}`)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleDeleteUser(email) {
    if (!window.confirm(`Delete ${email}?`)) return
    setError('')
    try {
      await adminDeleteUser(email)
      await load()
      setInfo(`Deleted ${email}`)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleUploadDataset(e) {
    e.preventDefault()
    if (!uploadFile) return
    setError('')
    setUploadingDataset(true)
    try {
      const result = await adminUploadDataset(uploadFile)
      setUploadFile(null)
      await load()
      setInfo(`Dataset uploaded: ${result.dataset_name}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploadingDataset(false)
    }
  }

  async function handleFetchDataset(datasetName, force = false) {
    setError('')
    setFetchingDataset(datasetName)
    if (force) {
      setInfo(`Re-downloading ${datasetName} from Hugging Face — this can take a few minutes…`)
    } else {
      setInfo(`Preparing ${datasetName} from Hugging Face…`)
    }
    try {
      const result = await adminFetchDataset(datasetName, { force })
      await load()
      const count = result.dataset?.records ?? result.dataset?.train_records ?? 0
      setInfo(`${datasetName} ready — ${count} records`)
    } catch (err) {
      setError(err.message)
      setInfo('')
    } finally {
      setFetchingDataset('')
    }
  }

  async function handleDownloadDataset(datasetName) {
    setError('')
    setInfo(datasetName === 'sample' ? 'Preparing the sample ZIP…' : `Preparing ${datasetName}.zip…`)
    setDownloadingDataset(datasetName)
    try {
      await adminDownloadDataset(datasetName)
      setInfo(`${datasetName}.zip downloaded`)
    } catch (err) {
      setError(err.message)
      setInfo('')
    } finally {
      setDownloadingDataset('')
    }
  }

  async function startTraining(modelType, datasetName) {
    const ds = datasetName
      ? datasets.find((d) => d.name === datasetName)
      : (datasets.find((d) => d.name !== 'sample') || datasets[0])
    if (!ds) {
      setError('No dataset available. Download or upload a dataset first.')
      return
    }
    setError('')
    const epochs = training[modelType]?.epochs
      || (modelType === 'legalbert' ? 2 : modelType === 'rgcn' ? 30 : modelType === 'bilstm_crf' ? 8 : undefined)

    try {
      const res = await adminTrainModel(modelType, ds.name, epochs)
      if (!res.ok) {
        throw new Error(`Training request failed (${res.status})`)
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      setTraining((t) => ({ ...t, [modelType]: { status: 'running', progress: 0, message: 'Starting…' } }))

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const payload = line.slice(6)
            try {
              const data = JSON.parse(payload)
              setTraining((t) => ({
                ...t,
                [modelType]: {
                  status: data.status || 'running',
                  progress: data.progress || 0,
                  message: data.message || '',
                  ...(data.result && { result: data.result }),
                },
              }))
              if (data.status === 'completed') {
                setTrainResult(data.result)
                setInfo(`${MODEL_TYPES.find(m => m.key === modelType)?.label} training complete!`)
                await load()
              }
              if (data.status === 'failed') {
                setError(data.message || 'Training failed')
              }
            } catch { /* ignore parse errors on partial SSE */ }
          }
        }
      }
    } catch (err) {
      setError(err.message)
      setTraining((t) => ({ ...t, [modelType]: { status: 'failed', progress: 0, message: err.message } }))
    }
  }

  async function activateModel(modelType, modelPath) {
    setError('')
    try {
      await adminActivateModel(modelType, modelPath)
      setInfo(`Activated ${modelType} model`)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  function handleLogout() {
    logout()
    navigate('/login')
  }

  if (loading) return <p className="text-textsoft text-sm">Loading admin panel…</p>

  return (
    <div className="space-y-6">
      <Eyebrow>Admin — Backend Management</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">Admin Control Panel</h1>
      <p className="text-textsoft text-sm mb-6">
        Manage accounts, datasets, and train AI models. Only administrators can access this page.
      </p>

      <Card className="p-5 mb-6">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="font-serif font-semibold text-ink">{me?.name}</div>
            <div className="text-[11px] font-mono text-textsoft">{me?.email}</div>
          </div>
          <Badge tone="conflict">ADMIN</Badge>
        </div>
        <div className="flex gap-4 mb-4">
          <Button onClick={() => load()} size="sm">Refresh</Button>
          <Button variant="ghost" onClick={handleLogout} size="sm">Sign out</Button>
        </div>
        {error && <p className="text-conflict text-xs mt-3">{error}</p>}
        {info && <p className="text-verified text-xs mt-1">{info}</p>}
      </Card>

      <div className="flex gap-1 border-b border-line mb-6 no-print">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            className={`text-sm px-4 py-2.5 border-b-2 -mb-px transition-colors ${
              activeTab === t.key ? 'border-gold text-ink font-semibold' : 'border-transparent text-textsoft hover:text-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === 'dashboard' && <DashboardTab me={me} users={users} datasets={datasets} models={models} training={training} />}
      {activeTab === 'users' && <UsersTab users={users} me={me} newUser={newUser} updateUser={updateUser} handleCreateUser={handleCreateUser} handleToggleRole={handleToggleRole} handleResetPassword={handleResetPassword} handleDeleteUser={handleDeleteUser} />}
      {activeTab === 'datasets' && <DatasetsTab datasets={datasets} uploadFile={uploadFile} setUploadFile={setUploadFile} uploadingDataset={uploadingDataset} handleUploadDataset={handleUploadDataset} catalog={catalog} fetchingDataset={fetchingDataset} handleFetchDataset={handleFetchDataset} downloadingDataset={downloadingDataset} handleDownloadDataset={handleDownloadDataset} />}
      {activeTab === 'training' && <TrainingTab modelTypes={MODEL_TYPES} datasets={datasets} training={training} setTraining={setTraining} startTraining={startTraining} trainResult={trainResult} />}
      {activeTab === 'models' && <ModelsTab models={models} activateModel={activateModel} />}
    </div>
  )
}

function DashboardTab({ me, users, datasets, models, training }) {
  const modelCount = Object.values(models).flat().length
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase text-textsoft mb-1">Total Users</div>
          <div className="font-serif text-2xl font-semibold text-ink">{users.length}</div>
        </Card>
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase text-textsoft mb-1">Datasets</div>
          <div className="font-serif text-2xl font-semibold text-ink">{datasets.length}</div>
        </Card>
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase text-textsoft mb-1">Trained Models</div>
          <div className="font-serif text-2xl font-semibold text-ink">{modelCount}</div>
        </Card>
      </div>

      <Card className="p-6">
        <h3 className="font-serif font-semibold text-ink mb-3">Quick Start</h3>
        <p className="text-textsoft text-sm mb-4">
          1. <strong>Download or upload a dataset</strong> in the Datasets tab.<br />
          2. <strong>Train a model</strong> on that dataset in the Model Training tab.<br />
          3. <strong>Activate</strong> the trained model in the Deployed Models tab so the website uses it.
        </p>
        <div className="space-y-3">
          {MODEL_TYPES.map((m) => (
            <div key={m.key} className="flex items-center justify-between p-3 bg-paperdim/50 rounded border border-line">
              <div>
                <div className="font-semibold text-ink">{m.label}</div>
                <div className="text-xs text-textsoft">{m.desc}</div>
              </div>
              <Badge tone="court">{training[m.key]?.status || 'Idle'}</Badge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function UsersTab({ users, me, newUser, updateUser, handleCreateUser, handleToggleRole, handleResetPassword, handleDeleteUser }) {
  return (
    <div className="space-y-6">
      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Create new account</h3>
        <form onSubmit={handleCreateUser} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <Input
            label="Full name"
            value={newUser.name}
            onChange={updateUser('name')}
            required
          />
          <Input
            label="Email"
            type="email"
            value={newUser.email}
            onChange={updateUser('email')}
            required
          />
          <Input
            label="Password"
            type="password"
            value={newUser.password}
            onChange={updateUser('password')}
            required
          />
          <div className="flex items-center">
            <input
              type="checkbox"
              checked={newUser.isAdmin}
              onChange={updateUser('isAdmin')}
              className="accent-gold"
            />
            <span className="ml-2 text-sm text-textsoft">Make administrator</span>
          </div>
          <Button type="submit" className="md:col-span-1">Create account</Button>
        </form>
      </Card>

      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">All accounts ({users.length})</h3>
        <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-line">
              {['Name', 'Email', 'Role', 'Action'].map((h) => (
                <th key={h} className="text-left text-[11px] uppercase tracking-wide text-textsoft font-semibold px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.email} className="border-b border-line last:border-none">
                <td className="px-4 py-3 text-sm text-ink">{u.name}</td>
                <td className="px-4 py-3 text-sm font-mono text-textsoft">{u.email}</td>
                <td className="px-4 py-3">
                  <Badge tone={u.is_admin ? 'conflict' : 'verified'}>
                    {u.is_admin ? 'Administrator' : 'User'}
                  </Badge>
                  {!u.verified && (
                    <span className="ml-2 font-mono text-[10px] text-conflict">unverified</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-col items-start gap-1">
                    <button
                      onClick={() => handleResetPassword(u.email)}
                      className="text-xs font-mono text-court hover:underline"
                    >
                      Reset password
                    </button>
                    {u.email !== me?.email && (
                      <>
                        <button
                          onClick={() => handleToggleRole(u.email, u.is_admin)}
                          className="text-xs font-mono text-gold hover:underline"
                        >
                          {u.is_admin ? 'Remove admin' : 'Make admin'}
                        </button>
                        <button
                          onClick={() => handleDeleteUser(u.email)}
                          className="text-xs font-mono text-conflict hover:underline"
                        >
                          Delete
                        </button>
                      </>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr><td colSpan={4} className="px-4 py-6 text-center text-sm text-textsoft">No accounts.</td></tr>
            )}
          </tbody>
        </table>
        </div>
      </Card>
    </div>
  )
}

function DatasetsTab({ datasets, uploadFile, setUploadFile, uploadingDataset, handleUploadDataset, catalog, fetchingDataset, handleFetchDataset, downloadingDataset, handleDownloadDataset }) {
  const PUBLIC_DATASETS = {
    legal_ner_large: {
      title: 'Indian Legal NER Corpus',
      detail: '9,600+ annotated Indian court judgments · 14 legal entity types · MIT licence',
    },
    legal_nli_large: {
      title: 'Balanced NLI Corpus (SNLI)',
      detail: '15,000 train + 3,000 test pairs · consistent / contradiction / neutral · CC-BY-SA',
    },
  }

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Download Large Public Datasets</h3>
        <p className="text-textsoft text-sm mb-4">
          One click downloads the datasets straight from Hugging Face and prepares them for training.
          These are the datasets that produced the high-accuracy models.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(catalog.length ? catalog : Object.keys(PUBLIC_DATASETS).map((name) => ({ name, built: false }))).map((item) => {
            const info = PUBLIC_DATASETS[item.name] || { title: item.name, detail: '' }
            const busy = fetchingDataset === item.name
            return (
              <div key={item.name} className="border border-line rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-ink">{info.title}</div>
                  <Badge tone={item.built ? 'verified' : 'gold'}>{item.built ? 'Ready' : 'Not downloaded'}</Badge>
                </div>
                <div className="text-xs text-textsoft mb-3">{info.detail}</div>
                <Button
                  size="sm"
                  onClick={() => handleFetchDataset(item.name, item.built)}
                  disabled={busy}
                >
                  {busy
                    ? (item.built ? 'Re-downloading…' : 'Downloading…')
                    : item.built ? '↻ Re-download' : '⬇ Download & Prepare'}
                </Button>
              </div>
            )
          })}
        </div>
      </Card>

      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Download Sample Dataset</h3>
        <p className="text-textsoft text-sm mb-4">
          Get started with the built-in MECR sample dataset containing 10 legal case documents
          with entity annotations (COURT, DEFENDANT, EVIDENCE, DATE, WITNESS, LAWYER, PLACE).
        </p>
        <Button
          onClick={() => handleDownloadDataset('sample')}
          disabled={downloadingDataset === 'sample'}
          className="w-full"
        >
          {downloadingDataset === 'sample' ? 'Preparing sample ZIP…' : '⬇ Download Sample MECR Dataset (ZIP)'}
        </Button>
        <p className="text-[11px] text-textsoft mt-2">
          Extract the ZIP and re-upload it, or upload your own ZIP with nested folders containing
          .txt, .json, .jsonl, .jpg, .png, or .pdf files.
        </p>
      </Card>

      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Upload Dataset (ZIP)</h3>
        <p className="text-textsoft text-sm mb-3">
          Upload a ZIP file containing JSON or text files. Supported formats:<br />
          • Label Studio: <code className="text-xs text-conflict">{"data.text + annotations"}</code><br />
          • Direct: <code className="text-xs text-conflict">{"{text, entities}"}</code><br />
          • Tuple: <code className="text-xs text-conflict">{"(text, {entities})"}</code><br />
          • Plain text: <code className="text-xs text-conflict">{"*.txt"}</code>
        </p>
        <form onSubmit={handleUploadDataset} className="flex gap-3 items-end">
          <input
            type="file"
            accept=".zip"
            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            className="hidden"
          />
          <Button
            type="button"
            onClick={() => document.querySelector('input[type=file]').click()}
            disabled={uploadingDataset}
          >
            Select ZIP file
          </Button>
          {uploadFile && <span className="text-sm font-mono text-textsoft truncate max-w-xs">{uploadFile.name}</span>}
          <Button type="submit" disabled={uploadingDataset || !uploadFile}>
            {uploadingDataset ? 'Uploading…' : 'Upload Dataset'}
          </Button>
        </form>
        {uploadingDataset && (
          <div className="mt-3">
            <div className="h-1.5 w-full bg-paperdim rounded-full overflow-hidden">
              <div className="h-full bg-gold" />
            </div>
          </div>
        )}
      </Card>

      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Available Datasets ({datasets.length})</h3>
        <table className="w-full">
          <thead>
            <tr className="border-b border-line">
              {['Name', 'Text Files', 'Images', 'JSON', 'Action'].map((h) => (
                <th key={h} className="text-left text-[11px] uppercase tracking-wide text-textsoft font-semibold px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.name} className="border-b border-line last:border-none">
                <td className="px-4 py-3 text-sm text-ink">{d.name}</td>
                <td className="px-4 py-3 text-sm text-textsoft font-mono">{d.text_files || 0}</td>
                <td className="px-4 py-3 text-sm text-textsoft font-mono">{d.image_files || 0}</td>
                <td className="px-4 py-3 text-sm text-textsoft font-mono">{d.json_files || 0}</td>
                <td className="px-4 py-3 text-right">
                  <Button
                    onClick={() => handleDownloadDataset(d.name)}
                    disabled={downloadingDataset === d.name}
                    size="sm"
                    variant="ghost"
                  >
                    {downloadingDataset === d.name ? 'Preparing…' : '⬇ Download'}
                  </Button>
                </td>
              </tr>
            ))}
            {datasets.length === 0 && (
              <tr><td colSpan={5} className="px-4 py-6 text-center text-sm text-textsoft">No datasets yet. Download the sample dataset above.</td></tr>
            )}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function TrainingTab({ modelTypes, datasets, training, setTraining, startTraining, trainResult }) {
  // LegalBERT needs premise/hypothesis pairs; the others need NER text.
  const datasetOptions = (modelKey) => {
    const filtered = modelKey === 'legalbert'
      ? datasets.filter((d) => d.name === 'legal_claims_large' || d.name === 'sample')
      : datasets.filter((d) => d.name !== 'legal_claims_large' && d.name !== 'legal_nli_large')
    return filtered.length ? filtered : datasets
  }

  const defaultDataset = (modelKey) => {
    const options = datasetOptions(modelKey)
    const preferred = modelKey === 'legalbert' ? 'legal_claims_large' : 'inlegalner_large'
    return options.find((d) => d.name === preferred)?.name || options[0]?.name || ''
  }

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <h3 className="font-serif font-semibold text-ink mb-3">Train a Model</h3>
        <p className="text-textsoft text-sm mb-4">
          Select a model type and dataset. Training runs in the backend — watch progress below.
          The large public datasets give the strongest accuracy.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {modelTypes.map((m) => {
            const t = training[m.key]
            const options = datasetOptions(m.key)
            const selectedDataset = t?.dataset || defaultDataset(m.key)
            return (
              <Card key={m.key} className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="font-semibold text-ink">{m.label}</div>
                  <Badge tone="court">{t?.status || 'Idle'}</Badge>
                </div>
                <div className="text-xs text-textsoft mb-3">{m.desc}</div>
                <div className="mb-2">
                  <label className="text-[11px] font-mono text-textsoft block mb-1">Dataset</label>
                  <select
                    className="w-full border border-line rounded-md px-2 py-1.5 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-gold"
                    value={selectedDataset}
                    onChange={(e) => {
                      setTraining((tr) => ({
                        ...tr,
                        [m.key]: { ...tr[m.key], dataset: e.target.value },
                      }))
                    }}
                    disabled={t?.status === 'running'}
                  >
                    {options.map((d) => (
                      <option key={d.name} value={d.name}>{d.name}</option>
                    ))}
                  </select>
                </div>
                <Button
                  onClick={() => startTraining(m.key, selectedDataset)}
                  disabled={t?.status === 'running' || options.length === 0}
                >
                  {t?.status === 'running' ? 'Training…' : 'Train Model'}
                </Button>
                {t && t.status === 'running' && (
                  <div className="mt-3">
                    <div className="flex justify-between text-xs text-textsoft mb-1">
                      <span>{t.message}</span>
                      <span>{t.progress}%</span>
                    </div>
                    <div className="h-1.5 bg-paperdim rounded-full overflow-hidden">
                      <div className={`h-full bg-gold`} style={{ width: `${t.progress}%` }} />
                    </div>
                  </div>
                )}
                {t && t.status === 'completed' && (
                  <div className="mt-3 p-3 bg-verifiedsoft/50 rounded border border-verified/30">
                    <div className="text-xs text-verified font-semibold mb-1">Training Complete ✓</div>
                    <div className="text-[11px] text-textsoft">
                      Accuracy: {t.result?.accuracy ?? 'N/A'}%
                      {t.result?.entity_f1 != null && ` · Entity F1: ${t.result.entity_f1}`}
                      {t.result?.f1 != null && t.result?.entity_f1 == null && ` · F1: ${t.result.f1}`}
                      {t.result?.roc_auc != null && ` · AUC: ${t.result.roc_auc}`}
                      {t.result?.samples != null && ` · Samples: ${t.result.samples}`}
                    </div>
                  </div>
                )}
                {t && t.status === 'failed' && (
                  <div className="mt-3 p-3 bg-conflictsoft/50 rounded border border-conflict/30">
                    <div className="text-xs text-conflict font-semibold mb-1">Failed</div>
                    <div className="text-[11px] text-textsoft">{t.message}</div>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      </Card>

      {trainResult && (
        <Card className="p-5">
          <h3 className="font-serif font-semibold text-ink mb-3">Latest Training Result</h3>
          <pre className="bg-paperdim p-4 rounded text-[11px] overflow-x-auto font-mono">
            {JSON.stringify(trainResult, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  )
}

function ModelsTab({ models, activateModel }) {
  return (
    <Card className="p-6">
      <h3 className="font-serif font-semibold text-ink mb-3">Model Versions &amp; Deployment</h3>
      <p className="text-textsoft text-sm mb-4">
        View all trained models and activate the one the website should use.
      </p>

      <div className="space-y-4">
        {Object.entries(models).map(([modelType, versions]) => (
          <div key={modelType} className="border border-line rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <div className="font-serif font-semibold text-ink">
                {modelTypesLabel(modelType)}
              </div>
              <Badge tone="gold">{versions.length} version(s)</Badge>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                <tr className="border-b border-line">
                  {['Dataset', 'Accuracy', 'Extra metric', '5-fold CV', 'Samples', 'Status'].map((h) => (
                    <th key={h} className="text-left text-[11px] uppercase tracking-wide text-textsoft font-semibold px-3 py-2">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {versions.map((v, i) => (
                  <tr key={i} className="border-b border-b border-line/50 last:border-none">
                    <td className="px-3 py-2 text-[11px] text-textsoft">{v.dataset ?? '—'}</td>
                    <td className="px-3 py-2 font-mono">{v.accuracy ?? 'N/A'}%</td>
                    <td className="px-3 py-2 font-mono text-[11px] text-textsoft">
                      {v.entity_f1 != null ? `F1 ${v.entity_f1}` :
                        v.f1 != null ? `F1 ${v.f1}` :
                        v.roc_auc != null ? `AUC ${v.roc_auc}` : '—'}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-textsoft" title={v.cv?.evaluation || ''}>
                      {v.cv
                        ? `${v.cv.accuracy_mean}% ± ${v.cv.accuracy_std} (F1 ${v.cv.entity_f1_mean})`
                        : '—'}
                    </td>
                    <td className="px-3 py-2 text-textsoft">{v.samples ?? 'N/A'}</td>
                    <td className="px-3 py-2 whitespace-nowrap">
                      {v.active ? (
                        <Badge tone="verified">Active</Badge>
                      ) : (
                        <button
                          onClick={() => activateModel(modelType, v.model_path)}
                          className="text-xs font-mono text-gold hover:underline"
                        >
                          Activate
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </div>
        ))}
        {Object.keys(models).length === 0 && (
          <p className="text-sm text-textsoft">No trained models yet. Train one in the Model Training tab.</p>
        )}
      </div>
    </Card>
  )
}

function modelTypesLabel(key) {
  const m = MODEL_TYPES.find(x => x.key === key)
  return m ? m.label : key
}
