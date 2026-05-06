import React from 'react';

const STATS = [
  { val: '97.6%', label: 'Best AUC-ROC', color: 'var(--mstf)' },
  { val: '3',     label: 'Models Trained', color: 'var(--blue)' },
  { val: '0.70',  label: 'Threshold', color: 'var(--purple)' },
  { val: '10',    label: 'Epochs', color: 'var(--gold)' },
  { val: 'FF++',  label: 'Dataset', color: 'var(--muted)' },
  { val: '3-Stream', label: 'MSTF-Trans', color: 'var(--mstf)' },
];

export default function Hero({ onStart }) {
  return (
    <div style={{ textAlign: 'center', padding: '48px 0 56px' }}>

      {/* Badge */}
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 8,
        background: 'rgba(0,223,162,0.07)', border: '1px solid rgba(0,223,162,0.2)',
        padding: '7px 18px', borderRadius: 50,
        fontSize: '0.78rem', color: 'rgba(0,223,162,0.9)', fontFamily: 'DM Mono',
        marginBottom: 28,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--mstf)', display: 'inline-block', boxShadow: '0 0 8px var(--mstf)', animation: 'pulse 2s infinite' }} />
        · SDB | DSFN | MSTF-Trans
        <style>{`@keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.3} }`}</style>
      </div>

      <h1 style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '3.6rem', letterSpacing: '-2px', lineHeight: 1.05, marginBottom: 20 }}>
        Detect{' '}
        <span style={{ background: 'linear-gradient(135deg, #ff3d5a 0%, #9b6bff 50%, #00dfa2 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
          Deepfakes
        </span>
        <br />With Multi-Stream AI
      </h1>

      <p style={{ fontSize: '1.05rem', color: 'var(--muted)', maxWidth: 580, margin: '0 auto 44px', lineHeight: 1.75 }}>
        Three custom-trained models — <strong style={{ color: 'var(--sdb)' }}>SDB</strong>, <strong style={{ color: 'var(--dsfn)' }}>DSFN</strong>, and <strong style={{ color: 'var(--mstf)' }}>MSTF-Trans</strong> — fusing spatial, frequency &amp; temporal streams with adaptive gating for state-of-the-art deepfake detection.
      </p>

      {/* Stats bar */}
      <div style={{
        display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 0,
        background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18,
        marginBottom: 0, overflow: 'hidden',
      }}>
        {STATS.map((s, i) => (
          <React.Fragment key={s.label}>
            <div style={{ padding: '22px 28px', textAlign: 'center', flex: '1 1 100px', minWidth: 100 }}>
              <div style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '1.5rem', color: s.color }}>{s.val}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--muted)', marginTop: 4, fontFamily: 'DM Mono', letterSpacing: 0.5 }}>{s.label}</div>
            </div>
            {i < STATS.length - 1 && <div style={{ width: 1, background: 'var(--border)', alignSelf: 'stretch' }} />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
