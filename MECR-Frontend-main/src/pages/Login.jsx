import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { login, getUser } from '../lib/api'
import { Button, Input } from '../components/ui/Primitives'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(true)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const justReset = location.state?.reset

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login({ email, password, remember })
      const user = getUser()
      navigate(user?.is_admin ? '/admin' : '/')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-paper rounded-lg border border-line p-8">
        <div className="font-serif text-2xl font-semibold text-ink mb-1">MECR</div>
        <div className="text-textsoft text-sm mb-7">Sign in to the case workbench</div>

        {justReset && <p className="text-verified text-xs mb-4">Password updated. Please sign in.</p>}

        <form onSubmit={handleSubmit}>
          <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@court.gov.in" required />
          <Input label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required />

          <div className="flex items-center justify-between mb-4 -mt-1">
            <label className="flex items-center gap-2 text-xs text-textsoft cursor-pointer">
              <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} className="accent-gold" />
              Remember me
            </label>
            <Link to="/forgot-password" className="text-xs text-gold font-semibold">Forgot password?</Link>
          </div>

          {error && <p className="text-conflict text-xs mb-4">{error}</p>}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? 'Signing in…' : 'Sign in'}
          </Button>
        </form>

        <p className="text-xs text-textsoft mt-5 text-center">
          New here? You must <Link to="/register" className="text-gold font-semibold">register an account</Link> first.
        </p>
      </div>
    </div>
  )
}
