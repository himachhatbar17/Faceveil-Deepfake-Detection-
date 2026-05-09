import React, { useState } from 'react';

const MODELS_DETAIL = [
  {
    id: 'SDB', dot: 'var(--sdb)', badge: 'BASELINE',
    full: 'Spatial-Domain Baseline',
    params: '~6.5M',
    backbone: 'EfficientNet-B0',
    loss: 'Label Smooth BCE (ε=0.1)',
    optimizer: 'AdamW (lr=3e-4, wd=1e-4)',
    scheduler: 'Warmup Cosine (3 warmup epochs)',
    streams: [
      { name: 'RGB Spatial Branch', color: 'var(--sdb)', desc: 'EfficientNet-B0 pre-trained on ImageNet. GlobalAvgPool removes spatial dims, then FC(256)→GELU→Dropout→FC(1) produces logit.' },
    ],
    novelty: 'Serves as the spatial-only baseline. No frequency or temporal information is used. Establishes a lower bound for the ablation study.',
    auc: 87.2, acc: 85.6, f1: 84.9, eer: 14.1, acer: 13.8,
  },
  {
    id: 'DSFN', dot: 'var(--dsfn)', badge: 'FAST',
    full: 'Dual-Branch Spatio-Frequency Network',
    params: '~8.2M',
    backbone: 'EfficientNet-B0 + FrequencyEncoder CNN',
    loss: 'Label Smooth BCE (ε=0.1)',
    optimizer: 'AdamW (lr=3e-4, wd=1e-4)',
    scheduler: 'Warmup Cosine (3 warmup epochs)',
    streams: [
      { name: 'RGB Spatial Branch', color: 'var(--blue)', desc: 'EfficientNet-B0 backbone → projected to feat_dim=256 via Linear+LayerNorm+GELU.' },
      { name: 'FFT Frequency Branch', color: 'var(--purple)', desc: 'CNN over 56×56 FFT magnitude maps. Conv2d(1→32)→BN→ReLU×2 layers → AdaptiveAvgPool → FC(256).' },
    ],
    fusion: 'CrossAttentionFusion — spatial features attend over frequency features via multi-head cross-attention, then concatenate and project to 256-d.',
    novelty: 'Adds a dedicated frequency stream. Frequency artifacts from GAN upsampling are invisible in spatial domain but clearly visible in FFT maps. Cross-attention lets spatial stream query frequency for relevant manipulation traces.',
    auc: 93.1, acc: 91.4, f1: 90.8, eer: 8.7, acer: 8.2,
  },
  {
    id: 'MSTF-Trans', dot: 'var(--mstf)', badge: '★ PROPOSED',
    full: 'Multi-Stream Temporal Fusion Transformer',
    params: '~22.4M',
    backbone: 'EfficientNet-B4 + FreqCNN + TemporalFlowEncoder',
    loss: 'Label Smooth BCE (ε=0.1)',
    optimizer: 'AdamW (lr=3e-4, wd=1e-4)',
    scheduler: 'Warmup Cosine (3 warmup epochs)',
    streams: [
      { name: 'RGB Spatial Branch', color: 'var(--blue)', desc: 'EfficientNet-B4 (larger backbone). Projected to 256-d via Linear+LayerNorm+GELU. Captures fine-grained texture inconsistencies.' },
      { name: 'FFT Frequency Branch', color: 'var(--purple)', desc: 'Same CNN encoder as DSFN over 56×56 FFT maps. Projects to 256-d.' },
      { name: 'Optical Flow Branch', color: 'var(--mstf)', desc: 'TemporalFlowEncoder: Conv2d(1→32)→BN→ReLU → AdaptiveAvgPool → FC(256). Operates on 56×56 optical flow magnitude maps.' },
    ],
    fusion: 'AdaptiveGatedFusion (core novelty) — learns per-sample softmax gates over 3 stream embeddings, producing weighted sum. Then TransformerEncoder (heads=4, depth=4, dim=256, FFN_dim=512, dropout=0.1) refines the fused representation. Final FC(256→1) sigmoid.',
    novelty: 'Adaptive gating allows the model to up-weight whichever stream is most discriminative for a given sample — e.g., weight temporal stream heavily for videos, frequency stream for GAN artifacts. Transformer captures inter-stream interactions that fixed fusion misses.',
    auc: 97.6, acc: 96.3, f1: 95.8, eer: 3.1, acer: 2.8,
  },
];

export default function Models() {
  const [active, setActive] = useState(2);
  const m = MODELS_DETAIL[active];

  return (
    <div>
      {/* Model Tabs */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 28 }}>
        {MODELS_DETAIL.map((mod, i) => (
          <button key={mod.id} onClick={() => setActive(i)} style={{
            flex: 1, padding: '18px 20px', borderRadius: 14, cursor: 'pointer', textAlign: 'left',
            background: active === i ? 'var(--ink2)' : 'var(--ink3)',
            border: `1px solid ${active === i ? mod.dot + '55' : 'var(--border)'}`,
            borderTop: active === i ? `3px solid ${mod.dot}` : '3px solid transparent',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontFamily: 'DM Mono', fontWeight: 600, fontSize: '0.9rem', color: active === i ? mod.dot : 'var(--muted)' }}>{mod.id}</span>
              <span style={{ fontSize: '0.65rem', color: mod.dot, fontFamily: 'DM Mono' }}>{mod.badge}</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--muted)', lineHeight: 1.4 }}>{mod.full}</div>
            <div style={{ marginTop: 10, display: 'flex', gap: 10 }}>
              {[['AUC', mod.auc + '%'], ['Acc', mod.acc + '%']].map(([k, v]) => (
                <span key={k} style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: active === i ? mod.dot : 'var(--dim)' }}>{v} <span style={{ color: 'var(--muted)', fontWeight: 400 }}>{k}</span></span>
              ))}
            </div>
          </button>
        ))}
      </div>

      {/* Detail Panel */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* Left: Architecture */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18, padding: 24 }}>
            <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 16 }}>INPUT STREAMS</div>
            {m.streams.map((s, i) => (
              <div key={s.name} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < m.streams.length - 1 ? '1px solid var(--border)' : 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: s.color, boxShadow: `0 0 8px ${s.color}` }} />
                  <span style={{ fontFamily: 'DM Mono', fontSize: '0.82rem', fontWeight: 600, color: s.color }}>{s.name}</span>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--muted)', lineHeight: 1.6 }}>{s.desc}</p>
              </div>
            ))}
          </div>

          {m.fusion && (
            <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18, padding: 24 }}>
              <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 10 }}>FUSION STRATEGY</div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text)', lineHeight: 1.7 }}>{m.fusion}</p>
            </div>
          )}

          <div style={{ background: 'var(--ink2)', border: `1px solid ${m.dot}30`, borderRadius: 18, padding: 24, borderLeft: `3px solid ${m.dot}` }}>
            <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 10 }}>KEY NOVELTY</div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text)', lineHeight: 1.7 }}>{m.novelty}</p>
          </div>
        </div>

        {/* Right: Config + Metrics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18, padding: 24 }}>
            <div style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 1, marginBottom: 16 }}>TRAINING CONFIG</div>
            {[
              ['Backbone', m.backbone],
              ['Parameters', m.params],
              ['Loss Function', m.loss],
              ['Optimizer', m.optimizer],
              ['LR Scheduler', m.scheduler],
              ['Batch Size', '32'],
              ['Epochs', '10'],
              ['Image Size', '224 × 224'],
              ['FFT Size', '56 × 56'],
              ['Flow Size', '56 × 56'],
              ['Threshold', '0.70'],
              ['Seed', '42'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--muted)', fontFamily: 'DM Mono', flexShrink: 0, marginRight: 12 }}>{k}</span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text)', fontFamily: 'DM Mono', textAlign: 'right' }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Metric boxes */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            {[['AUC-ROC', m.auc + '%', m.dot], ['Accuracy', m.acc + '%', 'var(--blue)'], ['F1 Score', m.f1 + '%', 'var(--purple)'], ['EER', m.eer + '%', 'var(--gold)'], ['ACER', m.acer + '%', 'var(--gold)'], ['Params', m.params, 'var(--muted)']].map(([k, v, c]) => (
              <div key={k} style={{ background: 'var(--ink3)', border: '1px solid var(--border)', borderRadius: 12, padding: '14px 12px', textAlign: 'center' }}>
                <div style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '1.3rem', color: c }}>{v}</div>
                <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginTop: 3 }}>{k}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
