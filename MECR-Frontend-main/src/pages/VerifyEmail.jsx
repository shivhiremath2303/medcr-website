import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { resendVerification, verifyEmail } from '../lib/api'
import { Button, Input } from '../components/ui/Primitives'

export default function VerifyEmail() {
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState(location.state?.email || '')
  const [token, setToken] = useState(location.state?.devToken || '')
  const [error, setError] = useState('')
  const [info, setInfo] = useState(
    location.state?.devToken ? 'Development mode: your code is pre-filled below.' : ''
  )
  const [loading, setLoading] = useState(false)

  async function handleVerify(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await verifyEmail({ email, token })
      navigate('/')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleResend() {
    setError('')
    setInfo('')
    try {
      const data = await resendVerification({ email })
      setInfo(`New code sent (dev): ${data.dev_verify_token}`)
      setToken(data.dev_verify_token)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-sm bg-paper rounded-lg border border-line p-8">
        <div className="font-serif text-2xl font-semibold text-ink mb-1">Verify your email</div>
        <div className="text-textsoft text-sm mb-7">Enter the 8-character code we sent to confirm your account.</div>

        <form onSubmit={handleVerify}>
          <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@court.gov.in" required />
          <Input label="Verification code" value={token} onChange={(e) => setToken(e.target.value)} placeholder="A1B2C3D4" required />
          {info && <p className="text-gold text-xs mb-3">{info}</p>}
          {error && <p className="text-conflict text-xs mb-4">{error}</p>}
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? 'Verifying…' : 'Verify email'}
          </Button>
        </form>

        <div className="flex justify-between mt-5">
          <button type="button" onClick={handleResend} className="text-xs text-gold font-semibold">Resend code</button>
          <Link to="/login" className="text-xs text-textsoft">Back to sign in</Link>
        </div>
      </div>
    </div>
  )
}
