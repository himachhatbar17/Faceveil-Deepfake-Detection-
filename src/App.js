import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Detector from './components/Detector';
import Models from './components/Models';
import Ablation from './components/Ablation';
import Results from './components/Results';

const PAGE_TITLES = {
  Detector: { title: 'Detection Engine', sub: 'Upload media and run inference with SDB, DSFN, or MSTF-Trans' },
  Models:   { title: 'Model Architecture', sub: 'Detailed breakdown of all 3 trained models and their components' },
  Ablation: { title: 'Ablation Study', sub: 'Stream contribution analysis — what each branch adds to performance' },
  Results:  { title: 'Performance Results', sub: 'ROC curves, training dynamics, confusion matrix and final metrics' },
};

export default function App() {
  const [activePage, setActivePage] = useState('Detector');

  const info = PAGE_TITLES[activePage];

  return (
    <div style={{ minHeight: '100vh', background: 'var(--ink)', position: 'relative' }}>

      {/* Ambient background glows */}
      <div style={{ position: 'fixed', top: -200, left: -200, width: 600, height: 600, borderRadius: '50%', background: 'radial-gradient(circle, rgba(255,61,90,0.05) 0%, transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />
      <div style={{ position: 'fixed', top: -100, right: -100, width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,120,255,0.05) 0%, transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />
      <div style={{ position: 'fixed', bottom: -100, left: '40%', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,223,162,0.04) 0%, transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />

      <Navbar activePage={activePage} setActivePage={setActivePage} />

      <div style={{ position: 'relative', zIndex: 1, maxWidth: 1320, margin: '0 auto', padding: '80px 40px 60px' }}>

        {/* Hero only on Detector */}
        {activePage === 'Detector' && <Hero />}

        {/* Page header for non-detector pages */}
        {activePage !== 'Detector' && (
          <div style={{ paddingTop: 20, marginBottom: 32 }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginBottom: 14, background: 'rgba(130,140,220,0.07)', border: '1px solid rgba(130,140,220,0.14)', padding: '5px 14px', borderRadius: 20 }}>
              <span style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)' }}>FACEVEIL · {activePage.toUpperCase()}</span>
            </div>
            <h2 style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '2.2rem', letterSpacing: '-1px', marginBottom: 8 }}>{info.title}</h2>
            <p style={{ fontSize: '0.9rem', color: 'var(--muted)' }}>{info.sub}</p>
          </div>
        )}

        {/* Pages */}
        {activePage === 'Detector' && <Detector />}
        {activePage === 'Models'   && <Models />}
        {activePage === 'Ablation' && <Ablation />}
        {activePage === 'Results'  && <Results />}
      </div>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '28px 40px', textAlign: 'center', color: 'var(--muted)', fontSize: '0.8rem', fontFamily: 'DM Mono', position: 'relative', zIndex: 1 }}>
        <span style={{ color: 'var(--text)' }}>FaceVeil</span> · DeepFake Detection Research ·{' '}
        <span style={{ color: 'var(--mstf)' }}>SDB</span> | <span style={{ color: 'var(--dsfn)' }}>DSFN</span> | <span style={{ color: 'var(--mstf)' }}>MSTF-Trans</span> ·{' '}
        FaceForensics++ , DFDC - selfconstructed Dataset · 
      </footer>
    </div>
  );
}
