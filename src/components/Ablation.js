import React from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts';

const ABLATION_VARIANTS = [
  { name: 'MSTF-Trans\n(Full)', short: 'Full', auc: 97.6, acc: 96.3, f1: 95.8, eer: 3.1, color: 'var(--mstf)', note: 'All 3 streams + Transformer + Adaptive Gate' },
  { name: 'w/o Temporal\n(S+F)', short: 'S+F', auc: 93.5, acc: 91.8, f1: 91.2, eer: 7.9, color: 'var(--dsfn)', note: 'Spatial + Frequency only, no optical flow branch' },
  { name: 'w/o Frequency\n(S+T)', short: 'S+T', auc: 90.2, acc: 88.7, f1: 88.1, eer: 11.2, color: 'var(--gold)', note: 'Spatial + Temporal only, no FFT branch' },
  { name: 'w/o Spatial\n(F+T)', short: 'F+T', auc: 88.4, acc: 87.0, f1: 86.6, eer: 13.4, color: 'var(--purple)', note: 'Frequency + Temporal only, no RGB backbone' },
  { name: 'DSFN\n(Baseline)', short: 'DSFN', auc: 93.1, acc: 91.4, f1: 90.8, eer: 8.7, color: '#4a78ff', note: 'Dual-branch without temporal stream' },
  { name: 'SDB\n(Spatial Only)', short: 'SDB', auc: 87.2, acc: 85.6, f1: 84.9, eer: 14.1, color: 'var(--sdb)', note: 'Pure spatial baseline' },
];

const RADAR_DATA = [
  { metric: 'AUC', Full: 97.6, 'S+F': 93.5, 'S+T': 90.2, SDB: 87.2 },
  { metric: 'Accuracy', Full: 96.3, 'S+F': 91.8, 'S+T': 88.7, SDB: 85.6 },
  { metric: 'F1', Full: 95.8, 'S+F': 91.2, 'S+T': 88.1, SDB: 84.9 },
  { metric: '100-EER', Full: 96.9, 'S+F': 92.1, 'S+T': 88.8, SDB: 85.9 },
  { metric: 'AP', Full: 97.1, 'S+F': 93.0, 'S+T': 89.5, SDB: 86.0 },
];

const BAR_DATA = ABLATION_VARIANTS.map(v => ({ name: v.short, AUC: v.auc, Acc: v.acc, F1: v.f1 }));

export default function Ablation() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16 }}>
        {ABLATION_VARIANTS.map(v => (
          <div key={v.short} style={{
            background: 'var(--ink2)', border: `1px solid var(--border)`,
            borderTop: `3px solid ${v.color}`,
            borderRadius: 16, padding: 22, transition: 'transform 0.2s',
          }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-3px)'}
            onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <span style={{ fontFamily: 'DM Mono', fontWeight: 700, fontSize: '0.88rem', color: v.color }}>{v.short}</span>
              <span style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '1.2rem', color: v.color }}>{v.auc}%</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--muted)', lineHeight: 1.5, marginBottom: 14 }}>{v.note}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[['AUC', v.auc], ['Accuracy', v.acc], ['F1', v.f1], ['EER', v.eer]].map(([k, val]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>{k}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 60, height: 4, background: 'var(--ink4)', borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${val}%`, background: v.color, borderRadius: 2 }} />
                    </div>
                    <span style={{ fontSize: '0.72rem', fontFamily: 'DM Mono', color: 'var(--text)', width: 36, textAlign: 'right' }}>{val}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* Bar Chart */}
        <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18, padding: 24 }}>
          <div style={{ fontFamily: 'Syne', fontWeight: 700, fontSize: '0.95rem', marginBottom: 4 }}>Metric Comparison</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: 20 }}>AUC, Accuracy, F1 across all variants</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={BAR_DATA} barCategoryGap="30%" barGap={3}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(130,140,220,0.07)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
              <YAxis domain={[80, 100]} tick={{ fill: 'var(--muted)', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'var(--ink3)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', fontFamily: 'DM Mono', fontSize: 12 }} />
              <Bar dataKey="AUC" fill="#00dfa2" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Acc" fill="#4a78ff" radius={[3, 3, 0, 0]} />
              <Bar dataKey="F1" fill="#9b6bff" radius={[3, 3, 0, 0]} />
              <Legend wrapperStyle={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'DM Mono' }} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18, padding: 24 }}>
          <div style={{ fontFamily: 'Syne', fontWeight: 700, fontSize: '0.95rem', marginBottom: 4 }}>Radar: Stream Contribution</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: 20 }}>Full vs ablated variants vs SDB baseline</div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="rgba(130,140,220,0.12)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--muted)', fontSize: 11, fontFamily: 'DM Mono' }} />
              <Radar name="Full" dataKey="Full" stroke="#00dfa2" fill="#00dfa2" fillOpacity={0.12} strokeWidth={2} />
              <Radar name="S+F" dataKey="S+F" stroke="#4a78ff" fill="#4a78ff" fillOpacity={0.08} strokeWidth={1.5} />
              <Radar name="S+T" dataKey="S+T" stroke="#ffb84d" fill="#ffb84d" fillOpacity={0.05} strokeWidth={1.5} />
              <Radar name="SDB" dataKey="SDB" stroke="#ff3d5a" fill="#ff3d5a" fillOpacity={0.05} strokeWidth={1} strokeDasharray="4 3" />
              <Tooltip contentStyle={{ background: 'var(--ink3)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)', fontFamily: 'DM Mono', fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'DM Mono' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table */}
      <div style={{ background: 'var(--ink2)', border: '1px solid var(--border)', borderRadius: 18, overflow: 'hidden' }}>
        <div style={{ padding: '18px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontFamily: 'Syne', fontWeight: 700 }}>Full Ablation Table</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--muted)', fontFamily: 'DM Mono' }}>FaceForensics++ Test Set · Threshold = 0.70</div>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: 'rgba(130,140,220,0.04)' }}>
              {['Variant', 'Streams', 'AUC-ROC', 'Accuracy', 'F1 Score', 'EER', 'Notes'].map(h => (
                <th key={h} style={{ padding: '12px 20px', textAlign: 'left', fontSize: '0.7rem', fontFamily: 'DM Mono', color: 'var(--muted)', letterSpacing: 0.8, fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ABLATION_VARIANTS.map((v, i) => (
              <tr key={v.short} style={{ borderTop: '1px solid var(--border)', background: i === 0 ? 'rgba(0,223,162,0.04)' : 'transparent' }}>
                <td style={{ padding: '14px 20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: v.color, flexShrink: 0 }} />
                    <span style={{ fontFamily: 'DM Mono', fontWeight: 600, fontSize: '0.82rem', color: v.color }}>{v.short}</span>
                    {i === 0 && <span style={{ fontSize: '0.62rem', color: 'var(--mstf)', fontFamily: 'DM Mono', background: 'rgba(0,223,162,0.1)', border: '1px solid rgba(0,223,162,0.25)', padding: '1px 7px', borderRadius: 10 }}>◄ BEST</span>}
                  </div>
                </td>
                <td style={{ padding: '14px 20px', fontSize: '0.75rem', color: 'var(--muted)', fontFamily: 'DM Mono' }}>
                  {v.short === 'Full' ? 'S+F+T' : v.short === 'DSFN' ? 'S+F' : v.short === 'SDB' ? 'S' : v.short}
                </td>
                {[v.auc, v.acc, v.f1, v.eer].map((val, j) => (
                  <td key={j} style={{ padding: '14px 20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontFamily: 'DM Mono', fontWeight: 700, fontSize: '0.85rem', color: i === 0 ? v.color : 'var(--text)' }}>{val}%</span>
                      <div style={{ width: 50, height: 3, background: 'rgba(130,140,220,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${val}%`, background: v.color, borderRadius: 2 }} />
                      </div>
                    </div>
                  </td>
                ))}
                <td style={{ padding: '14px 20px', fontSize: '0.74rem', color: 'var(--muted)' }}>{v.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
