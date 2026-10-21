import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { logout, getUser, isAdmin } from '../lib/api'

const NAV = [
  { to: '/', label: 'Dashboard', num: '01' },
  { to: '/cases/new', label: 'New Case', num: '02' },
  { to: '/forgery-check', label: 'Forgery Screen', num: '03' }
]

export default function AppShell() {
  const navigate = useNavigate()
  const user = getUser()

  function handleLogout() {
    logout()
    navigate('/login')
  }

  const adminLinks = isAdmin()
    ? [
        { to: '/admin', label: 'Admin Panel', num: '04' }
      ]
    : []

  return (
    <div className="grid min-h-screen" style={{ gridTemplateColumns: '230px 1fr' }}>
      <aside className="bg-ink text-paper px-5 py-7 border-r border-[#223050] flex flex-col no-print">
        <div className="font-serif text-[19px] font-semibold">MECR</div>
        <div className="font-mono text-[11px] text-[#8A93A8] tracking-wide mb-8">MULTI-EVIDENCE CONFLICT RESOLVER</div>

        <nav className="flex-1">
          {[...NAV, ...adminLinks].map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2.5 rounded-md text-sm mb-1 transition-colors ${
                  isActive ? 'bg-gold text-ink font-semibold' : 'text-[#B7BECD] hover:bg-[#1C2740] hover:text-paper'
                }`
              }
            >
              <span className="font-mono text-[11px] w-4 opacity-70">{item.num}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <button onClick={handleLogout} className="text-left text-xs font-mono text-[#5D6883] hover:text-paper pt-4 border-t border-[#223050]">
          Sign out
        </button>
      </aside>

      <main className="bg-paper px-11 py-9 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
