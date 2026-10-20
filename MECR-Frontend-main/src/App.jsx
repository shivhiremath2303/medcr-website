import { Navigate, Route, Routes } from 'react-router-dom'
import AppShell from './components/AppShell'
import Login from './pages/Login'
import Register from './pages/Register'
import VerifyEmail from './pages/VerifyEmail'
import ForgotPassword from './pages/ForgotPassword'
import Dashboard from './pages/Dashboard'
import CaseCreate from './pages/CaseCreate'
import CaseDetail from './pages/CaseDetail'
import CaseUpload from './pages/CaseUpload'
import CaseAnalysis from './pages/CaseAnalysis'
import AdminPage from './pages/AdminPage'
import { getToken, isAdmin } from './lib/api'

function isAuthed() {
  return Boolean(getToken())
}

function RequireAuth({ children }) {
  return isAuthed() ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/verify-email" element={<VerifyEmail />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />

      <Route
        path="/"
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="cases/new" element={<CaseCreate />} />
        <Route path="cases/:caseId" element={<CaseDetail />} />
        <Route path="cases/:caseId/upload" element={<CaseUpload />} />
        <Route path="cases/:caseId/analysis" element={<CaseAnalysis />} />
        <Route path="admin" element={<AdminPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
