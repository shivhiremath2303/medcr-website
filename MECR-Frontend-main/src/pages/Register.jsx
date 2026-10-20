import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { register } from '../lib/api'
import { Button, Input, PasswordChecklist } from '../components/ui/Primitives'
import { EMAIL_RE, PASSWORD_RULES } from '../lib/passwordRules'

export default function Register() {
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  function update(field) {
    return (e) => setForm((f) => ({ ...f, [field]: e.target.value }))
  }

  const passwordValid = PASSWORD_RULES.every((r) => r.test(form.password))
  const emailValid = EMAIL_RE.test(form.email)
  const nameValid = form.name.trim().length > 0
  const passwordsMatch = form.confirm.length > 0 && form.password === form.confirm
  const canSubmit = nameValid && emailValid && passwordValid && passwordsMatch

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    if (!canSubmit) return
    setLoading(true)
    try {
      const data = await register({ name: form.name, email: form.email, password: form.password })
      navigate('/verify-email', { state: { email: form.email, devToken: data.dev_verify_token } })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-sm bg-paper rounded-lg border border-line p-8">
        <div className="font-serif text-2xl font-semibold text-ink mb-1">Create account</div>
        <div className="text-textsoft text-sm mb-7">For legal professionals and investigators</div>

        <form onSubmit={handleSubmit}>
          <Input label="Full name" value={form.name} onChange={update('name')} placeholder="Adv. Priya Sharma" required />
          <Input label="Email" type="email" value={form.email} onChange={update('email')} placeholder="you@court.gov.in" required />
          {form.email && !emailValid && (
            <p className="text-conflict text-xs -mt-3 mb-3">Enter a valid email address</p>
          )}

          <Input label="Password" type="password" value={form.password} onChange={update('password')} placeholder="••••••••" required />
          <PasswordChecklist password={form.password} rules={PASSWORD_RULES} />

          <Input label="Confirm password" type="password" value={form.confirm} onChange={update('confirm')} placeholder="••••••••" required />
          {form.confirm && !passwordsMatch && (
            <p className="text-conflict text-xs -mt-3 mb-3">Passwords do not match</p>
          )}

          {error && <p className="text-conflict text-xs mb-4">{error}</p>}
          <Button type="submit" className="w-full" disabled={loading || !canSubmit}>
            {loading ? 'Creating account…' : 'Create account'}
          </Button>
        </form>

        <p className="text-xs text-textsoft mt-5 text-center">
          Already registered? <Link to="/login" className="text-gold font-semibold">Sign in</Link>
        </p>
      </div>
    </div>
  )
}
