// -----------------------------------------------------------------------
// API layer — wired to the FastAPI backend (http://localhost:8000/api/v1)
// Every network call lives here; components never call fetch directly.
// -----------------------------------------------------------------------

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

// --- Token storage (supports "remember me") -----------------------------

export function getToken() {
  return localStorage.getItem('mecr_token') || sessionStorage.getItem('mecr_token')
}

export function saveToken(token, remember) {
  if (remember) {
    localStorage.setItem('mecr_token', token)
    sessionStorage.removeItem('mecr_token')
  } else {
    sessionStorage.setItem('mecr_token', token)
    localStorage.removeItem('mecr_token')
  }
}

// --- User storage (role + profile) ----------------------------------------

export function saveUser(user) {
  const payload = JSON.stringify(user)
  if (localStorage.getItem('mecr_token')) {
    localStorage.setItem('mecr_user', payload)
  } else {
    sessionStorage.setItem('mecr_user', payload)
  }
}

export function getUser() {
  const raw = localStorage.getItem('mecr_user') || sessionStorage.getItem('mecr_user')
  return raw ? JSON.parse(raw) : null
}

export function isAdmin() {
  const u = getUser()
  return Boolean(u && u.is_admin)
}

function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function json(res) {
  const text = await res.text()
  let data = {}
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = { detail: text }
    }
  }
  if (res.status === 401) {
    // Session expired (e.g. the backend restarted and in-memory tokens were
    // lost). Drop the stale token and send the user to the sign-in page
    // instead of stranding them on a dead screen.
    logout()
    if (!window.location.pathname.startsWith('/login')) {
      window.location.assign('/login')
    }
  }
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
  return data
}

// --- Auth ---------------------------------------------------------------

export async function login({ email, password, remember = false }) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  const data = await json(res)
  saveToken(data.access_token, remember)
  saveUser(data.user)
  return data
}

export async function register({ name, email, password }) {
  const res = await fetch(`${BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  })
  return json(res)
}

export async function verifyEmail({ email, token }) {
  const res = await fetch(`${BASE_URL}/auth/verify-email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, token }),
  })
  const data = await json(res)
  saveToken(data.access_token, false)
  saveUser(data.user)
  return data
}

export async function resendVerification({ email }) {
  const res = await fetch(`${BASE_URL}/auth/resend-verification`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  })
  return json(res)
}

export async function forgotPassword({ email }) {
  const res = await fetch(`${BASE_URL}/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  })
  return json(res)
}

export async function resetPassword({ token, password }) {
  const res = await fetch(`${BASE_URL}/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, password }),
  })
  return json(res)
}

export function logout() {
  localStorage.removeItem('mecr_token')
  sessionStorage.removeItem('mecr_token')
  localStorage.removeItem('mecr_user')
  sessionStorage.removeItem('mecr_user')
}

// --- Cases ----------------------------------------------------------------

export async function listCases() {
  const res = await fetch(`${BASE_URL}/cases`, { headers: authHeaders() })
  return json(res)
}

export async function createCase({ caseId, ipcSections, court, defendants }) {
  const res = await fetch(`${BASE_URL}/cases`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ caseId, ipcSections, court, defendants }),
  })
  return json(res)
}

export async function getCase(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}`, { headers: authHeaders() })
  return json(res)
}

export async function deleteCase(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}`, { method: 'DELETE', headers: authHeaders() })
  return json(res)
}

// --- Evidence -------------------------------------------------------------

export async function uploadEvidence(caseId, files, onProgress) {
  const formData = new FormData()
  for (const f of files) formData.append('files', f)
  const res = await fetch(`${BASE_URL}/cases/${caseId}/evidence`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })
  const data = await json(res)
  onProgress?.(100)
  return data
}

export async function listEvidence(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/evidence`, { headers: authHeaders() })
  return json(res)
}

export async function tagEvidence(evidenceId, personId) {
  const res = await fetch(`${BASE_URL}/evidence/${evidenceId}/link`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ evidenceId, personId }),
  })
  return json(res)
}

export async function deleteEvidence(evidenceId) {
  const res = await fetch(`${BASE_URL}/evidence/${evidenceId}`, { method: 'DELETE', headers: authHeaders() })
  return json(res)
}

export async function getDocumentAnalysis(evidenceId) {
  const res = await fetch(`${BASE_URL}/evidence/${evidenceId}/document`, { headers: authHeaders() })
  return json(res)
}

// --- Forgery screening (secondary experimental feature) --------------------

/** Run forgery screening on an already-uploaded evidence item (image only). */
export async function checkForgery(evidenceId) {
  const res = await fetch(`${BASE_URL}/evidence/${evidenceId}/forgery-check`, {
    headers: authHeaders(),
  })
  return json(res)
}

/** Upload a fresh image directly for forgery screening (standalone tool). */
export async function checkForgeryDirect(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${BASE_URL}/forgery/check`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })
  return json(res)
}

// --- AI processing ----------------------------------------------------------

export async function runAnalysis(caseId, onStatus) {
  const controller = new AbortController()
  // Safety net: never leave the UI spinning forever if the stream stalls.
  const timer = setTimeout(() => controller.abort(), 300000)
  try {
    const res = await fetch(`${BASE_URL}/analysis/run/${caseId}`, {
      method: 'POST',
      headers: authHeaders(),
      signal: controller.signal,
    })
    if (!res.ok) {
      let detail = `Analysis failed (${res.status})`
      try {
        const err = await res.json()
        if (err.detail) detail = err.detail
      } catch (_) {}
      throw new Error(detail)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let complete = false
    try {
      while (!complete) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (!payload) continue
          if (payload === '__complete__') {
            complete = true
            break
          }
          if (payload.startsWith('__error__')) {
            throw new Error(payload.slice('__error__'.length).trim() || 'Analysis failed')
          }
          onStatus?.(payload)
        }
      }
    } finally {
      // Release the socket as soon as we are done, even if the server is
      // still holding the response open.
      try { await reader.cancel() } catch (_) {}
    }
    return { caseId, status: 'complete' }
  } finally {
    clearTimeout(timer)
  }
}

// --- Analysis results (Module 3) --------------------------------------------

export async function getConflictMatrix(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/conflicts`, { headers: authHeaders() })
  return json(res)
}

export async function getAnomalies(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/anomalies`, { headers: authHeaders() })
  return json(res)
}

export async function getGraph(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/graph`, { headers: authHeaders() })
  return json(res)
}

export async function getEntities(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/entities`, { headers: authHeaders() })
  return json(res)
}

export async function getParties(caseId) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/parties`, { headers: authHeaders() })
  return json(res)
}

export async function reviewFinding(caseId, findingId, status) {
  const res = await fetch(`${BASE_URL}/cases/${caseId}/review/${findingId}`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  })
  return json(res)
}

// --- Report (Module 4) ------------------------------------------------------

export async function reportMeta(caseId) {
  const res = await fetch(`${BASE_URL}/report/${caseId}`, { headers: authHeaders() })
  return json(res)
}

export async function getReportContent(caseId) {
  const res = await fetch(`${BASE_URL}/report/${caseId}/content`, { headers: authHeaders() })
  return json(res)
}

export async function regenerateReport(caseId) {
  const res = await fetch(`${BASE_URL}/report/${caseId}/regenerate`, {
    method: 'POST',
    headers: authHeaders(),
  })
  return json(res)
}

export async function downloadReportPdf(caseId, fallbackFilename = 'conflict-report.pdf') {
  const res = await fetch(`${BASE_URL}/report/${caseId}/pdf`, {
    headers: authHeaders(),
  })
  if (!res.ok) {
    let errMessage = 'Failed to generate PDF'
    try {
      const err = await res.json()
      if (err.detail) errMessage = err.detail
    } catch (_) {}
    throw new Error(errMessage)
  }
  const blob = await res.blob()
  let filename = fallbackFilename
  const disposition = res.headers.get('Content-Disposition')
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename="?([^";]+)"?/)
    if (match && match[1]) filename = match[1]
  }
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// --- Admin management (Admin only) ------------------------------------------

export async function adminMe() {
  const res = await fetch(`${BASE_URL}/admin/me`, { headers: authHeaders() })
  return json(res)
}

export async function adminListUsers() {
  const res = await fetch(`${BASE_URL}/admin/users`, { headers: authHeaders() })
  return json(res)
}

export async function adminCreateUser({ name, email, password, isAdmin }) {
  const res = await fetch(`${BASE_URL}/admin/users`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password, is_admin: isAdmin }),
  })
  return json(res)
}

export async function adminToggleRole(email, isAdmin) {
  const res = await fetch(`${BASE_URL}/admin/users/${encodeURIComponent(email)}/role`, {
    method: 'PUT',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ is_admin: isAdmin }),
  })
  return json(res)
}

export async function adminResetPassword(email, password) {
  const res = await fetch(`${BASE_URL}/admin/users/${encodeURIComponent(email)}/password`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })
  return json(res)
}

export async function adminDeleteUser(email) {
  const res = await fetch(`${BASE_URL}/admin/users/${encodeURIComponent(email)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  return json(res)
}

// --- Admin — Datasets & Model Training --------------------------------------

export async function adminUploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${BASE_URL}/admin/dataset/upload`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })
  return json(res)
}

export async function adminListDatasets() {
  const res = await fetch(`${BASE_URL}/admin/datasets`, { headers: authHeaders() })
  return json(res)
}

export async function adminDatasetCatalog() {
  const res = await fetch(`${BASE_URL}/admin/datasets/catalog`, { headers: authHeaders() })
  return json(res)
}

export async function adminFetchDataset(datasetName, { force = false } = {}) {
  const query = force ? '?force=true' : ''
  const res = await fetch(`${BASE_URL}/admin/dataset/fetch/${encodeURIComponent(datasetName)}${query}`, {
    method: 'POST',
    headers: authHeaders(),
  })
  return json(res)
}

export async function adminDownloadDataset(datasetName) {
  const res = await fetch(`${BASE_URL}/admin/dataset/download/${encodeURIComponent(datasetName)}`, {
    headers: authHeaders(),
  })
  if (!res.ok) {
    let detail = `Download failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // response was not JSON — keep the status message
    }
    throw new Error(detail)
  }
  const blob = await res.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${datasetName}.zip`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // Give the browser a moment to start the download before releasing the blob.
  setTimeout(() => window.URL.revokeObjectURL(url), 1000)
  return { downloaded: datasetName }
}

export async function adminTrainModel(modelType, datasetName, epochs) {
  const res = await fetch(`${BASE_URL}/admin/train/${modelType}`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_name: datasetName, epochs }),
  })
  return res
}

export async function adminTrainingStatus(jobId) {
  const res = await fetch(`${BASE_URL}/admin/training/status/${jobId}`, { headers: authHeaders() })
  return json(res)
}

export async function adminListModels() {
  const res = await fetch(`${BASE_URL}/admin/models`, { headers: authHeaders() })
  return json(res)
}

export async function adminActivateModel(modelType, modelPath) {
  const res = await fetch(`${BASE_URL}/admin/models/activate`, {
    method: 'POST',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_type: modelType, model_path: modelPath }),
  })
  return json(res)
}
