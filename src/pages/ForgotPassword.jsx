import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { forgotPassword, resetPassword } from '../lib/api'
import { Button, Input, PasswordChecklist } from '../components/ui/Primitives'
import { PASSWORD_RULES } from '../lib/passwordRules'

export default function ForgotPassword() {
  const navigate = useNavigate()
  const [step, setStep] = useState(1)
  const [email, setEmail] = useState('')
  const [token, setToken] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [loading, setLoading] = useState(false)

  const passwordValid = PASSWORD_RULES.every((r) => r.test(password))
  const match = confirm.length > 0 && password === confirm
  const canReset = passwordValid && match

  async function requestCode(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const data = await forgotPassword({ email })
      if (data.dev_reset_token) {
        setToken(data.dev_reset_token)
        setInfo(`Development mode: your reset code is ${data.dev_reset_token}`)
      } else {
        setInfo(data.message)
      }
      setStep(2)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function doReset(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await resetPassword({ token, password })
      navigate('/login', { state: { reset: true } })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-sm bg-paper rounded-lg border border-line p-8">
        <div className="font-serif text-2xl font-semibold text-ink mb-1">Reset password</div>
        <div className="text-textsoft text-sm mb-7">
          {step === 1 ? 'Enter your email to receive a reset code.' : 'Enter the code and choose a new password.'}
        </div>

        {step === 1 ? (
          <form onSubmit={requestCode}>
            <Input label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@court.gov.in" required />
            {error && <p className="text-conflict text-xs mb-4">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Sending…' : 'Send reset code'}
            </Button>
          </form>
        ) : (
          <form onSubmit={doReset}>
            {info && <p className="text-gold text-xs mb-3">{info}</p>}
            <Input label="Reset code" value={token} onChange={(e) => setToken(e.target.value)} placeholder="A1B2C3D4" required />
            <Input label="New password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required />
            <PasswordChecklist password={password} rules={PASSWORD_RULES} />
            <Input label="Confirm new password" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="••••••••" required />
            {confirm && !match && <p className="text-conflict text-xs -mt-3 mb-3">Passwords do not match</p>}
            {error && <p className="text-conflict text-xs mb-4">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading || !canReset}>
              {loading ? 'Updating…' : 'Update password'}
            </Button>
          </form>
        )}

        <p className="text-xs text-textsoft mt-5 text-center">
          Remembered it? <Link to="/login" className="text-gold font-semibold">Back to sign in</Link>
        </p>
      </div>
    </div>
  )
}
