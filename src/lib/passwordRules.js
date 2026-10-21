// Shared password / email validation rules (mirrors the backend).
export const PASSWORD_RULES = [
  { test: (p) => p.length >= 8, label: 'At least 8 characters' },
  { test: (p) => /[A-Z]/.test(p), label: 'One uppercase letter (A–Z)' },
  { test: (p) => /[a-z]/.test(p), label: 'One lowercase letter (a–z)' },
  { test: (p) => /\d/.test(p), label: 'One number (0–9)' },
  { test: (p) => /[^A-Za-z0-9]/.test(p), label: 'One special character (!@#$…)' }
]

export const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/
