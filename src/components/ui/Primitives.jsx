export function Button({ children, variant = 'solid', className = '', ...props }) {
  const base = 'font-sans font-semibold text-sm px-5 py-2.5 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed'
  const styles = {
    solid: 'bg-ink text-paper hover:bg-ink2',
    ghost: 'bg-transparent text-ink border border-line hover:bg-paperdim',
    danger: 'bg-conflict text-paper hover:opacity-90'
  }
  return (
    <button className={`${base} ${styles[variant]} ${className}`} {...props}>
      {children}
    </button>
  )
}

export function Input({ label, ...props }) {
  return (
    <label className="block mb-4">
      {label && <span className="block text-xs font-mono uppercase tracking-wide text-textsoft mb-1.5">{label}</span>}
      <input
        className="w-full border border-line rounded-md px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-gold"
        {...props}
      />
    </label>
  )
}

export function Card({ children, className = '' }) {
  return <div className={`bg-white border border-line rounded-lg ${className}`}>{children}</div>
}

export function Badge({ tone = 'court', children }) {
  const tones = {
    court: 'bg-courtsoft text-court',
    conflict: 'bg-conflictsoft text-conflict',
    verified: 'bg-verifiedsoft text-verified',
    gold: 'bg-goldsoft text-[#8A6A20]'
  }
  return (
    <span className={`inline-block font-mono text-[10.5px] font-semibold px-2 py-1 rounded-full ${tones[tone]}`}>
      {children}
    </span>
  )
}

export function ScoreBar({ score, label }) {
  const pct = Math.round(score * 100)
  const tone = pct >= 75 ? 'bg-verified' : pct >= 50 ? 'bg-gold' : 'bg-conflict'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-textsoft">{label}</span>
        <span className="font-mono font-semibold">{pct}%</span>
      </div>
      <div className="h-1.5 w-full bg-paperdim rounded-full overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function Eyebrow({ children }) {
  return <div className="font-mono text-[11px] tracking-widest uppercase text-gold mb-2">{children}</div>
}

export function PasswordChecklist({ password, rules }) {
  return (
    <ul className="mb-4 -mt-2 space-y-1">
      {rules.map((r) => {
        const ok = r.test(password)
        return (
          <li key={r.label} className={`text-[11px] flex items-center gap-1.5 ${ok ? 'text-verified' : 'text-textsoft'}`}>
            <span className="font-mono">{ok ? '✓' : '○'}</span> {r.label}
          </li>
        )
      })}
    </ul>
  )
}
