import { useRef, useState } from 'react'
import { checkForgeryDirect } from '../lib/api'
import { Button, Card, Eyebrow } from '../components/ui/Primitives'

// Map raw_result → display label, colour token, and icon
const RESULT_META = {
  'LIKELY AUTHENTIC': {
    label: 'Likely Authentic',
    color: 'text-verified',
    bg: 'bg-verifiedsoft border-verified/30',
    bar: 'bg-verified',
    icon: '✓',
  },
  'POTENTIALLY TAMPERED': {
    label: 'Potentially Tampered',
    color: 'text-conflict',
    bg: 'bg-conflictsoft border-conflict/30',
    bar: 'bg-conflict',
    icon: '⚠',
  },
  'REVIEW REQUIRED': {
    label: 'Review Required',
    color: 'text-gold',
    bg: 'bg-goldsoft border-gold/30',
    bar: 'bg-gold',
    icon: '?',
  },
}

function ResultBadge({ data }) {
  const meta = RESULT_META[data.raw_result] || RESULT_META['REVIEW REQUIRED']
  const pct = Math.round((data.tampered_score ?? 0) * 100)
  const authPct = 100 - pct

  return (
    <div className={`rounded-xl border p-6 ${meta.bg}`}>
      {/* Verdict row */}
      <div className="flex items-center gap-3 mb-4">
        <span className={`font-mono text-3xl font-bold ${meta.color}`}>{meta.icon}</span>
        <div>
          <div className={`font-serif text-2xl font-semibold ${meta.color}`}>{meta.label}</div>
          <div className="text-[11px] font-mono text-textsoft uppercase tracking-widest mt-0.5">
            Experimental screening result
          </div>
        </div>
      </div>

      {/* Score bars */}
      <div className="space-y-3 mb-5">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-textsoft font-mono">Tamper likelihood</span>
            <span className={`font-mono font-semibold ${meta.color}`}>{pct}%</span>
          </div>
          <div className="h-2 w-full bg-white/60 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${meta.bar}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-textsoft font-mono">Authenticity likelihood</span>
            <span className="font-mono font-semibold text-verified">{authPct}%</span>
          </div>
          <div className="h-2 w-full bg-white/60 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-verified transition-all duration-700"
              style={{ width: `${authPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Decision thresholds */}
      <div className="flex gap-4 text-[11px] font-mono text-textsoft mb-4">
        <span>Low threshold: {Math.round((data.low_threshold ?? 0.35) * 100)}%</span>
        <span>·</span>
        <span>High threshold: {Math.round((data.high_threshold ?? 0.65) * 100)}%</span>
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-textsoft border-l-2 border-gold pl-3 leading-5">
        {data.disclaimer ||
          'This is an experimental, secondary screening feature. Output is not guaranteed proof of authenticity or tampering.'}
      </p>
    </div>
  )
}

export default function ForgeryCheck() {
  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  function handleFileSelected(f) {
    if (!f) return

    const ALLOWED = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp']
    const isAllowedType = ALLOWED.includes(f.type) || /\.(jpg|jpeg|png|tiff?|bmp|webp)$/i.test(f.name)
    if (!isAllowedType) {
      setError('Only image files are accepted (JPEG, PNG, TIFF, BMP, WebP).')
      return
    }

    setFile(f)
    setResult(null)
    setError('')
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragActive(false)
    const f = e.dataTransfer.files?.[0]
    handleFileSelected(f)
  }

  async function handleScreen() {
    if (!file) return
    setLoading(true)
    setResult(null)
    setError('')
    try {
      const data = await checkForgeryDirect(file)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Screening failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="max-w-2xl">
      <Eyebrow>Experimental Feature · Forgery Screening</Eyebrow>
      <h1 className="font-serif text-3xl text-ink mb-1">Document Authenticity Screen</h1>
      <p className="text-textsoft text-sm mb-2 max-w-xl">
        Upload a document image to run an AI-based authenticity screen using Error Level Analysis
        (ELA) and visual feature extraction.
      </p>
      <div className="inline-flex items-center gap-1.5 bg-goldsoft border border-gold/25 rounded-md px-3 py-1.5 mb-7">
        <span className="text-[11px] font-mono text-[#8A6A20] uppercase tracking-wide font-semibold">
          Accuracy: 73.89% · ROC-AUC: 0.7954
        </span>
        <span className="text-[10px] text-textsoft font-mono">· Experimental — not legal proof</span>
      </div>

      {/* Drop zone */}
      {!file && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          className={`border-2 border-dashed rounded-xl px-6 py-14 text-center transition-colors cursor-pointer ${
            dragActive ? 'border-gold bg-goldsoft/40' : 'border-line bg-white hover:border-gold/60'
          }`}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
          aria-label="Upload image for forgery screening"
        >
          <div className="font-mono text-xs text-gold tracking-wide mb-2">
            .JPEG · .PNG · .TIFF · .BMP · .WebP
          </div>
          <h3 className="font-serif font-semibold text-ink mb-1">Drop image or click to browse</h3>
          <p className="text-textsoft text-xs mb-4">Upload the document scan you want to screen</p>
          <input
            ref={inputRef}
            id="forgery-file-input"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFileSelected(e.target.files?.[0])}
          />
          <Button variant="ghost" type="button" onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}>
            Select image
          </Button>
        </div>
      )}

      {/* Preview + actions */}
      {file && (
        <Card className="p-5">
          <div className="flex gap-5 items-start">
            {preview && (
              <img
                src={preview}
                alt="Document preview"
                className="w-32 h-40 object-contain rounded-md border border-line bg-paperdim shrink-0"
              />
            )}
            <div className="flex-1 min-w-0">
              <div className="font-semibold text-ink text-sm truncate mb-0.5">{file.name}</div>
              <div className="text-xs text-textsoft font-mono mb-4">
                {(file.size / 1024).toFixed(1)} KB · {file.type || 'image'}
              </div>
              <div className="flex gap-2">
                <Button
                  id="forgery-screen-btn"
                  onClick={handleScreen}
                  disabled={loading}
                >
                  {loading ? 'Screening…' : 'Run Screen'}
                </Button>
                <Button variant="ghost" onClick={handleReset} disabled={loading}>
                  Change file
                </Button>
              </div>
            </div>
          </div>

          {/* Spinner */}
          {loading && (
            <div className="mt-5 flex items-center gap-3">
              <div className="relative w-6 h-6 shrink-0">
                <div className="absolute inset-0 border-2 border-paperdim rounded-full" />
                <div className="absolute inset-0 border-2 border-gold border-t-transparent rounded-full animate-spin" />
              </div>
              <span className="text-xs text-textsoft font-mono">
                Running ELA + feature extraction…
              </span>
            </div>
          )}
        </Card>
      )}

      {/* Error */}
      {error && (
        <div
          id="forgery-error"
          className="mt-4 rounded-lg border border-conflict/30 bg-conflictsoft px-4 py-3 text-sm text-conflict"
        >
          <span className="font-semibold font-mono mr-2">Error:</span>{error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div id="forgery-result" className="mt-5">
          <ResultBadge data={result} />
        </div>
      )}

      {/* Info box */}
      <Card className="mt-7 p-5">
        <div className="text-[11px] font-mono uppercase tracking-widest text-textsoft mb-3">How it works</div>
        <div className="grid grid-cols-3 gap-4 text-xs text-textsoft">
          <div className="bg-paperdim/50 rounded-md p-3">
            <div className="font-semibold text-ink text-[11px] mb-1 font-mono uppercase tracking-wide">ELA</div>
            Compares JPEG re-compression artefacts across image regions to find editing discontinuities.
          </div>
          <div className="bg-paperdim/50 rounded-md p-3">
            <div className="font-semibold text-ink text-[11px] mb-1 font-mono uppercase tracking-wide">HOG Features</div>
            Histogram of Oriented Gradients captures texture patterns that differ between authentic and tampered zones.
          </div>
          <div className="bg-paperdim/50 rounded-md p-3">
            <div className="font-semibold text-ink text-[11px] mb-1 font-mono uppercase tracking-wide">Extra Trees</div>
            Ensemble classifier trained on document images; outputs a tamper probability score.
          </div>
        </div>

        <div className="mt-4 text-[11px] text-textsoft border-t border-line pt-3 flex flex-wrap gap-x-6 gap-y-1">
          <span><span className="text-verified font-semibold">●</span> Score ≤ 35% → Likely Authentic</span>
          <span><span className="text-gold font-semibold">●</span> 35–65% → Review Required</span>
          <span><span className="text-conflict font-semibold">●</span> Score ≥ 65% → Potentially Tampered</span>
        </div>
      </Card>
    </div>
  )
}
