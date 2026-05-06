import React, { useState, useRef } from 'react';

const MODELS = [
  {
    id: 'SDB',
    name: 'SDB',
    full: 'Spatial-Domain Baseline',
    arch: 'EfficientNet-B0 → GlobalAvgPool → FC(256) → FC(1)',
    desc: 'Spatial-only baseline using EfficientNet-B0 backbone. Processes RGB frames without frequency or temporal branches.',
    auc: 87.2, acc: 85.6, f1: 84.9,
    speed: '0.06s', badge: 'BASELINE', badgeColor: 'var(--muted)', dot: 'var(--sdb)',
    streams: ['RGB Spatial'],
    features: [
      { name: 'Spatial Texture', val: 78, color: 'var(--sdb)' },
      { name: 'Blending Artifacts', val: 54, color: 'var(--sdb)' },
      { name: 'Color Consistency', val: 61, color: 'var(--sdb)' },
      { name: 'Frequency Artifacts', val: 12, color: 'var(--sdb)' },
      { name: 'Temporal Coherence', val: 8, color: 'var(--sdb)' },
    ],
  },
  {
    id: 'DSFN',
    name: 'DSFN',
    full: 'Dual-Branch Spatio-Frequency Net',
    arch: 'EfficientNet-B0 (Spatial) + CNN (FFT) + CrossAttentionFusion',
    desc: 'Adds a CNN frequency encoder over FFT maps fused with spatial features via cross-attention. Better at compression & blending artifacts.',
    auc: 93.1, acc: 91.4, f1: 90.8,
    speed: '0.10s', badge: 'FAST', badgeColor: 'var(--blue)', dot: 'var(--dsfn)',
    streams: ['RGB Spatial', 'FFT Frequency'],
    features: [
      { name: 'Spatial Texture', val: 81, color: 'var(--dsfn)' },
      { name: 'Blending Artifacts', val: 74, color: 'var(--dsfn)' },
      { name: 'Color Consistency', val: 69, color: 'var(--dsfn)' },
      { name: 'Frequency Artifacts', val: 85, color: 'var(--dsfn)' },
      { name: 'Temporal Coherence', val: 15, color: 'var(--dsfn)' },
    ],
  },
  {
    id: 'MSTF-Trans',
    name: 'MSTF-Trans',
    full: 'Multi-Stream Temporal Fusion Transformer',
    arch: 'EfficientNet-B4 (Spatial) + CNN (FFT) + TemporalFlowEncoder + AdaptiveGatedFusion + Transformer(heads=4,depth=4)',
    desc: 'Full proposed model. Fuses spatial, frequency (FFT), and temporal (optical flow) streams via adaptive gating and transformer encoder.',
    auc: 97.6, acc: 96.3, f1: 95.8,
    speed: '0.18s', badge: '★ PROPOSED', badgeColor: 'var(--green)', dot: 'var(--mstf)',
    streams: ['RGB Spatial', 'FFT Frequency', 'Optical Flow'],
    features: [
      { name: 'Spatial Texture', val: 89, color: 'var(--mstf)' },
      { name: 'Blending Artifacts', val: 91, color: 'var(--mstf)' },
      { name: 'Color Consistency', val: 84, color: 'var(--mstf)' },
      { name: 'Frequency Artifacts', val: 93, color: 'var(--mstf)' },
      { name: 'Temporal Coherence', val: 88, color: 'var(--mstf)' },
    ],
  },
];

const MOCK_RESULTS = {
  fake: {
    score: 82,
    verdict: 'DEEPFAKE DETECTED',
    fakeProb: 82.4,
    realProb: 17.6,
    method: 'Face Swap',
    region: 'Face Boundary',
    frames: '24/30 Fake',
    compression: 'c23',
    color: 'var(--red)',
    icon: '⚠',
  },
  real: {
    score: 94,
    verdict: 'AUTHENTIC',
    fakeProb: 6.1,
    realProb: 93.9,
    method: 'None Detected',
    region: 'No Anomaly',
    frames: '0/30 Fake',
    compression: 'Original',
    color: 'var(--green)',
    icon: '✓',
  },
};

function ScoreRing({ score, color, label }) {
  const r = 66, circ = 2 * Math.PI * r;
  const offset = circ * (1 - score / 100);
  return (
    <div style={{ position: 'relative', width: 160, height: 160 }}>
      <svg viewBox="0 0 160 160" width="160" height="160" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="80" cy="80" r={r} fill="none" stroke="rgba(130,140,220,0.08)" strokeWidth="12" />
        <circle cx="80" cy="80" r={r} fill="none" stroke={color} strokeWidth="12"
          strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.3s' }} />
      </svg>
      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', textAlign: 'center' }}>
        <div style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '2.2rem', color, lineHeight: 1 }}>{score}%</div>
        <div style={{ fontSize: '0.62rem', color: 'var(--muted)', fontFamily: 'DM Mono', marginTop: 2, letterSpacing: 1 }}>{label}</div>
      </div>
    </div>
  );
}

function Heatmap() {
  const cells = Array.from({ length: 40 }, (_, i) => {
    const row = Math.floor(i / 8), col = i % 8;
    const hot = [10, 11, 12, 18, 19, 20, 26, 27].includes(i);
    const med = [9, 13, 17, 21, 25, 28].includes(i);
    let bg = hot ? `rgba(255,61,90,${0.7 + Math.random() * 0.25})`
           : med ? `rgba(255,150,60,${0.4 + Math.random() * 0.2})`
           : `rgba(74,120,255,${0.05 + Math.random() * 0.12})`;
    return { id: i, bg };
  });
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8,1fr)', gap: 3, padding: 8, height: 180, background: '#0a0b15', borderRadius: 10 }}>
      {cells.map(c => <div key={c.id} style={{ borderRadius: 4, background: c.bg, opacity: 0.9 }} />)}
    </div>
  );
}

export default function Detector() {
  const [selectedModel, setSelectedModel] = useState(2); // MSTF-Trans
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('Overview');
  const inputRef = useRef();

  const model = MODELS[selectedModel];

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
  };

  const handleAnalyze = () => {
    if (!file) return;
    setAnalyzing(true);
    setResult(null);
    setTimeout(() => {
      setAnalyzing(false);
      setResult(Math.random() > 0.45 ? MOCK_RESULTS.fake : MOCK_RESULTS.real);
    }, 2200);
  };

  return (
    <div>
      {/* ── Upload + Model ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 20, marginBottom: 24 }}>

        {/* Upload Card */}
        <div style={{ background: 'var(--ink2)', border: `1px solid var(--border)`, borderRadius: 20, padding: 32 }}>
          <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 18 }}>UPLOAD MEDIA</div>

          <div
            onClick={() => inputRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
            style={{
              border: `2px dashed ${dragging ? 'var(--blue)' : file ? 'var(--green)' : 'rgba(130,140,220,0.22)'}`,
              borderRadius: 14, padding: '48px 32px', textAlign: 'center', cursor: 'pointer',
              background: dragging ? 'rgba(74,120,255,0.04)' : file ? 'rgba(0,223,162,0.03)' : 'rgba(130,140,220,0.02)',
              transition: 'all 0.3s',
            }}>
            <input ref={inputRef} type="file" accept="image/*,video/*" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
            <div style={{ fontSize: '3rem', marginBottom: 14 }}>{file ? '🎞️' : '🎬'}</div>
            <div style={{ fontFamily: 'Syne', fontWeight: 700, fontSize: '1.05rem', marginBottom: 6 }}>
              {file ? file.name : 'Drop your file here'}
            </div>
            <div style={{ color: 'var(--muted)', fontSize: '0.83rem', marginBottom: 20 }}>
              {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready to analyze` : 'Upload an image or video to detect deepfake manipulation'}
            </div>
            <div style={{ display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap' }}>
              {['JPG', 'PNG', 'MP4', 'AVI', 'MOV', 'WEBM'].map(f => (
                <span key={f} style={{ background: 'rgba(130,140,220,0.08)', border: '1px solid rgba(130,140,220,0.15)', color: 'var(--dim)', padding: '3px 10px', borderRadius: 20, fontSize: '0.72rem', fontFamily: 'DM Mono' }}>{f}</span>
              ))}
            </div>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file || analyzing}
            style={{
              marginTop: 20, width: '100%', padding: '14px 0',
              background: file && !analyzing ? 'linear-gradient(135deg, var(--blue), var(--purple))' : 'rgba(130,140,220,0.08)',
              border: 'none', borderRadius: 12, cursor: file && !analyzing ? 'pointer' : 'not-allowed',
              color: file && !analyzing ? '#fff' : 'var(--dim)',
              fontFamily: 'Syne', fontWeight: 700, fontSize: '0.95rem', letterSpacing: 0.3,
              boxShadow: file && !analyzing ? '0 8px 24px rgba(74,120,255,0.25)' : 'none',
            }}>
            {analyzing ? '⏳  Analyzing...' : '⚡  Run Detection'}
          </button>

          {/* Config row */}
          <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
            {[
              { label: 'Threshold', val: '0.70' },
              { label: 'Img Size', val: '224×224' },
              { label: 'Epochs', val: '10' },
              { label: 'Batch', val: '32' },
            ].map(c => (
              <div key={c.label} style={{ flex: 1, background: 'var(--ink3)', borderRadius: 10, padding: '8px 10px', textAlign: 'center' }}>
                <div style={{ fontSize: '0.78rem', fontFamily: 'DM Mono', color: 'var(--text)', fontWeight: 500 }}>{c.val}</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--muted)', marginTop: 2 }}>{c.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Selector */}
        <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 20, padding: 24, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 4 }}>SELECT MODEL</div>

          {MODELS.map((m, i) => (
            <div key={m.id} onClick={() => setSelectedModel(i)} style={{
              background: selectedModel === i ? 'rgba(130,140,220,0.08)' : 'var(--ink3)',
              border: `1px solid ${selectedModel === i ? 'rgba(130,140,220,0.28)' : 'var(--border)'}`,
              borderRadius: 14, padding: 16, cursor: 'pointer',
              borderLeft: selectedModel === i ? `3px solid ${m.dot}` : `1px solid var(--border)`,
              transition: 'all 0.2s',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: m.dot, display: 'inline-block', boxShadow: selectedModel === i ? `0 0 8px ${m.dot}` : 'none' }} />
                  <span style={{ fontFamily: 'DM Mono', fontWeight: 500, fontSize: '0.88rem' }}>{m.name}</span>
                </div>
                <span style={{ fontSize: '0.68rem', color: m.badgeColor, fontFamily: 'DM Mono', fontWeight: 700 }}>{m.badge}</span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--muted)', lineHeight: 1.5, marginBottom: 10 }}>{m.desc.slice(0, 80)}…</div>
              <div style={{ display: 'flex', gap: 12 }}>
                {[['AUC', m.auc + '%'], ['Acc', m.acc + '%'], ['Speed', m.speed]].map(([k, v]) => (
                  <div key={k}>
                    <span style={{ fontFamily: 'DM Mono', fontWeight: 600, fontSize: '0.78rem', color: m.dot }}>{v}</span>
                    <span style={{ fontSize: '0.68rem', color: 'var(--muted)', marginLeft: 3 }}>{k}</span>
                  </div>
                ))}
              </div>
              {/* Stream badges */}
              <div style={{ display: 'flex', gap: 4, marginTop: 10, flexWrap: 'wrap' }}>
                {m.streams.map(s => (
                  <span key={s} style={{ fontSize: '0.65rem', fontFamily: 'DM Mono', background: 'rgba(130,140,220,0.08)', border: '1px solid rgba(130,140,220,0.14)', color: 'var(--dim)', padding: '2px 8px', borderRadius: 10 }}>{s}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Results ── */}
      {(result || analyzing) && (
        <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 20, padding: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <div>
              <div style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '1.3rem' }}>Detection Results</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 3 }}>
                {model.name} · {model.full} · {model.speed}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 4, background: 'var(--ink3)', border: '1px solid var(--border)', borderRadius: 10, padding: 4 }}>
              {['Overview', 'Heatmap', 'Features'].map(t => (
                <button key={t} onClick={() => setActiveTab(t)} style={{
                  padding: '6px 14px', borderRadius: 7, border: 'none', cursor: 'pointer', fontSize: '0.8rem', fontFamily: 'DM Sans', fontWeight: 600,
                  background: activeTab === t ? 'rgba(130,140,220,0.18)' : 'transparent',
                  color: activeTab === t ? 'var(--text)' : 'var(--muted)',
                }}>{t}</button>
              ))}
            </div>
          </div>

          {analyzing ? (
            <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--muted)' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: 16 }}>⚙️</div>
              <div style={{ fontFamily: 'DM Mono', fontSize: '0.9rem' }}>Running {model.name} inference…</div>
              <div style={{ marginTop: 20, height: 4, background: 'var(--ink3)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ height: '100%', background: 'linear-gradient(90deg,var(--blue),var(--purple))', borderRadius: 2, animation: 'slide 1.5s infinite', width: '40%' }} />
              </div>
              <style>{`@keyframes slide { 0%{margin-left:-40%} 100%{margin-left:100%} }`}</style>
            </div>
          ) : result && (
            <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr 260px', gap: 20 }}>

              {/* Score Ring */}
              <div style={{ background: 'var(--ink3)', borderRadius: 16, padding: 24, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
                <ScoreRing
                  score={result === MOCK_RESULTS.fake ? 82 : 94}
                  color={result.color}
                  label={result === MOCK_RESULTS.fake ? 'FAKE' : 'REAL'} />

                <div style={{ background: `${result.color}18`, border: `1px solid ${result.color}44`, color: result.color, padding: '8px 16px', borderRadius: 20, fontFamily: 'DM Mono', fontWeight: 700, fontSize: '0.78rem', textAlign: 'center' }}>
                  {result.icon} {result.verdict}
                </div>

                {[['Fake Probability', result.fakeProb, 'var(--red)'], ['Real Probability', result.realProb, 'var(--green)']].map(([label, val, color]) => (
                  <div key={label} style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 5 }}>
                      <span style={{ color: 'var(--muted)' }}>{label}</span>
                      <span style={{ color, fontFamily: 'DM Mono', fontWeight: 700 }}>{val}%</span>
                    </div>
                    <div style={{ height: 5, background: 'rgba(130,140,220,0.08)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${val}%`, background: color, borderRadius: 3, transition: 'width 1s ease' }} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Heatmap / Main panel */}
              <div style={{ background: 'var(--ink3)', borderRadius: 16, padding: 20 }}>
                {activeTab === 'Overview' || activeTab === 'Heatmap' ? (
                  <>
                    <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 12 }}>GradCAM MANIPULATION HEATMAP</div>
                    <Heatmap />
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 14 }}>
                      {[['Method', result.method], ['Region', result.region], ['Compression', result.compression], ['Frames', result.frames]].map(([k, v]) => (
                        <div key={k} style={{ background: `${result.color}0d`, border: `1px solid ${result.color}28`, borderRadius: 8, padding: '10px 12px' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginBottom: 3 }}>{k}</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, fontFamily: 'DM Mono', color: 'var(--text)' }}>{v}</div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 16 }}>FEATURE IMPORTANCE ANALYSIS</div>
                    {model.features.map(f => (
                      <div key={f.name} style={{ marginBottom: 14 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 5 }}>
                          <span style={{ color: 'var(--muted)' }}>{f.name}</span>
                          <span style={{ fontFamily: 'DM Mono', color: f.color, fontWeight: 600 }}>{f.val}%</span>
                        </div>
                        <div style={{ height: 5, background: 'rgba(130,140,220,0.08)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${f.val}%`, background: f.color, borderRadius: 3 }} />
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>

              {/* Model Architecture */}
              <div style={{ background: 'var(--ink3)', borderRadius: 16, padding: 20 }}>
                <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 14 }}>MODEL ARCHITECTURE</div>
                <div style={{ fontSize: '0.8rem', fontFamily: 'DM Mono', color: 'var(--text)', lineHeight: 1.7, background: 'var(--ink)', padding: 12, borderRadius: 8, border: '1px solid var(--border)', wordBreak: 'break-word' }}>
                  {model.arch.split(' + ').map((part, i) => (
                    <div key={i} style={{ marginBottom: 4 }}>
                      {i > 0 && <span style={{ color: 'var(--muted)' }}>↓ </span>}
                      <span style={{ color: i === 0 ? 'var(--blue)' : i === 1 ? 'var(--purple)' : 'var(--green)' }}>{part}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 10 }}>INPUT STREAMS</div>
                  {model.streams.map((s, i) => (
                    <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: ['var(--blue)', 'var(--purple)', 'var(--green)'][i] }} />
                      <span style={{ fontSize: '0.8rem', fontFamily: 'DM Mono', color: 'var(--text)' }}>{s}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 16, padding: 12, background: 'var(--ink)', borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginBottom: 6, fontFamily: 'DM Mono' }}>CONFIG</div>
                  {[['Threshold', '0.70'], ['feat_dim', '256'], ['tf_heads', '4'], ['tf_depth', '4']].map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                      <span style={{ color: 'var(--muted)', fontFamily: 'DM Mono' }}>{k}</span>
                      <span style={{ color: 'var(--text)', fontFamily: 'DM Mono', fontWeight: 500 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
