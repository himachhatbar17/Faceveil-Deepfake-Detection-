import React, { useState, useEffect } from 'react';

const NAV_LINKS = ['Detector', 'Models', 'Ablation', 'Results'];

export default function Navbar({ activePage, setActivePage }) {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
      height: 64,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 40px',
      background: scrolled ? 'rgba(11,12,19,0.92)' : 'rgba(11,12,19,0.6)',
      backdropFilter: 'blur(20px)',
      borderBottom: `1px solid ${scrolled ? 'rgba(130,140,220,0.14)' : 'transparent'}`,
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }} onClick={() => setActivePage('Detector')}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: 'linear-gradient(135deg, #ff3d5a, #9b6bff)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 15, fontWeight: 900,
        }}>F</div>
        <span style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '1.15rem', letterSpacing: '-0.5px' }}>
          Face<span style={{ background: 'linear-gradient(90deg,#ff3d5a,#9b6bff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Veil</span>
        </span>
        <span style={{ fontSize: '0.7rem', color: 'var(--muted)', fontFamily: 'DM Mono', marginTop: 2 }}>v1.0</span>
      </div>

      {/* Links */}
      <div style={{ display: 'flex', gap: 4 }}>
        {NAV_LINKS.map(link => (
          <button key={link} onClick={() => setActivePage(link)} style={{
            background: activePage === link ? 'rgba(130,140,220,0.12)' : 'transparent',
            border: activePage === link ? '1px solid rgba(130,140,220,0.22)' : '1px solid transparent',
            borderRadius: 8, padding: '6px 16px', cursor: 'pointer',
            color: activePage === link ? 'var(--text)' : 'var(--muted)',
            fontFamily: 'DM Sans', fontWeight: 500, fontSize: '0.85rem',
          }}>{link}</button>
        ))}
      </div>

      {/* Badge */}
      <div style={{
        background: 'rgba(0,223,162,0.1)', border: '1px solid rgba(0,223,162,0.25)',
        color: 'var(--green)', padding: '5px 14px', borderRadius: 20,
        fontSize: '0.75rem', fontFamily: 'DM Mono', fontWeight: 500,
        display: 'flex', alignItems: 'center', gap: 6,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block', boxShadow: '0 0 6px var(--green)' }} />
        RESEARCH WORK
      </div>
    </nav>
  );
}
