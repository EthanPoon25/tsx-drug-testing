import { useEffect, useMemo, useRef, useState } from "react";
import React from "react";
import "./styles/app.css";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from "recharts";

// Type definitions
interface Synapse {
  id: number
  strength: number
  active: boolean
  x: number
  y: number
  history: number[]
}

interface RewardData {
  epoch: number
  reward: number
  sd: number
}

interface BindingData {
  step: number
  energy: number
  convergence: number
}

interface Molecule {
  id: number
  structure: string
  bits: number[]
  density: number
  similarity: number
  energy: number
  contFp: number[]
  smiles?: string
}

interface BrainParams {
  plasticity: number
  noise: number
  enablePlasticity: boolean
}

interface SystemParams {
  plasticity: number
  noise: number
  enablePlasticity: boolean
  epochs: number
  batch: number
  lrGen: number
  lrDisc: number
  lambda: number
  seed: number
  modBrain: boolean
  modVqe: boolean
  modPlas: boolean
  modDisc: boolean
}

interface LogEntry {
  level: "info" | "warn" | "err" | "debug"
  msg: string
}

interface SystemStatus {
  running: boolean
  epoch: number
  progress: number
  ganLoss: number
  discLoss: number
  vqePenalty: number
  logs: LogEntry[]
}

interface Metrics {
  rewardMean: number
  rewardSd: number
  synAvg: number
  stateMag: number
}

interface User {
  email: string
}

interface HistoryEntry {
  ts: number
  summary: {
    rewardMean: number
    bindingSteps?: number
    epochs: number
    ganLoss: number
    discLoss?: number
    vqePenalty: number
  }
}

interface IconProps {
  name: string
  size?: number
}

interface ToggleProps {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
  tip?: string
}

interface ProgressProps {
  value: number
}

interface BrainPanelProps {
  synapses: Synapse[]
  rewards: RewardData[]
  params: BrainParams
  setParams: (params: (prev: BrainParams) => BrainParams) => void
  metrics: Metrics
}

interface MoleculesPanelProps {
  molecules: Molecule[]
  onRegenerate: () => void
  selected: number
  setSelected: (index: number) => void
  onExport: () => void
  onFileUpload: (message: string) => void
}

interface QuantumPanelProps {
  binding: BindingData[]
  vqeSteps: number
  setVqeSteps: (steps: number) => void
  onRerun: () => void
}

interface ControlsPanelProps {
  status: SystemStatus
  setStatus: React.Dispatch<React.SetStateAction<SystemStatus>>
  params: SystemParams
  setParams: React.Dispatch<React.SetStateAction<SystemParams>>
  onStartPause: () => void
  onReset: () => void
  onSave: () => void
  onLoad: (e: React.ChangeEvent<HTMLInputElement>) => void
}

interface LoginSignupProps {
  onLogin: (token: string, email: string) => void
  onSignup: (token: string, email: string) => void
  isLogin?: boolean
}

interface UserHistoryProps {
  history: HistoryEntry[]
  onLoadHistory: (entry: HistoryEntry) => void
}

interface FileUploadProps {
  onFileUpload: (message: string) => void
  onRegenerate: () => void
}

interface Tab {
  id: string
  label: string
  icon: string
}

// Data generation functions
const genSynapses = (): Synapse[] =>
  Array.from({ length: 64 }, (_, i) => {
    const strength = Math.random()
    return {
      id: i,
      strength,
      active: Math.random() > 0.7,
      x: i % 8,
      y: Math.floor(i / 8),
      history: Array.from({ length: 16 }, (_, h) => Math.max(0, Math.min(1, strength + (Math.sin((i + h) * 0.5) * 0.2) + (Math.random() - 0.5) * 0.1))),
    }
  })

const genRewards = (len: number = 120, offset: number = 0): RewardData[] =>
  Array.from({ length: len }, (_, i) => {
    const epoch = i + offset
    const base = 0.55 + Math.sin(epoch * 0.07) * 0.18 + (Math.random() - 0.5) * 0.06
    return { epoch, reward: Math.max(0, Math.min(1, base)), sd: 0.05 + Math.random() * 0.07 }
  })

const genBinding = (len: number = 70): BindingData[] =>
  Array.from({ length: len }, (_, i) => {
    const energy = -32 - Math.exp(-i * 0.07) * 18 + (Math.random() - 0.5) * 1.6
    const conv = Math.max(0, 100 - i * 1.2 + Math.random() * 4)
    return { step: i, energy, convergence: conv }
  })

const genMolecules = (n: number = 8): Molecule[] =>
  Array.from({ length: n }, (_, id) => {
    const bits: number[] = Array.from({ length: 16 }, () => (Math.random() > 0.52 ? 1 : 0));
  const on = bits.reduce((a, b) => a + b, 0);
  const similarity = 40 + Math.random() * 60;
    const energy = -25 - Math.random() * 20
    return {
      id,
      structure: `C${4 + Math.floor(Math.random() * 16)}H${8 + Math.floor(Math.random() * 30)}N${Math.floor(Math.random() * 4)}O${Math.floor(Math.random() * 6)}`,
      bits,
      density: on / bits.length,
      similarity,
      energy,
      contFp: bits.map(b => (b ? Math.min(1, Math.random() * 0.5 + 0.5) : Math.random() * 0.4)),
    }
  })

const Icon: React.FC<IconProps> = ({ name, size = 18 }) => {
  const stroke = "currentColor"
  const s = size
  const common = { width: s, height: s, viewBox: "0 0 24 24", fill: "none", stroke, strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const }
  
  switch (name) {
    case "brain":
      return (
        <svg {...common}><path d="M8 8a3 3 0 0 0-3 3v1a3 3 0 1 0 0 6h2"/><path d="M16 8a3 3 0 0 1 3 3v1a3 3 0 1 1 0 6h-2"/><path d="M8 8V5a3 3 0 0 1 6 0v3"/></svg>
      )
    case "atom":
      return (
        <svg {...common}><circle cx="12" cy="12" r="1.5"/><path d="M19 12c0 3.866-3.134 7-7 7S5 15.866 5 12 8.134 5 12 5s7 3.134 7 7Z"/><path d="M5.5 7.5c3.5 3.5 9.5 9.5 13 13"/></svg>
      )
    case "flask":
      return (
        <svg {...common}><path d="M9 3h6"/><path d="M10 3v5l-4.5 7.5A3 3 0 0 0 8 20h8a3 3 0 0 0 2.5-4.5L14 8V3"/></svg>
      )
    case "settings":
      return (
        <svg {...common}><path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V22a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06A2 2 0 0 1 3.8 19.4l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H2a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06A2 2 0 0 1 5.04 3.8l.06.06a1.65 1.65 0 0 0 1.82.33H7a1.65 1.65 0 0 0 1-1.51V2a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06A2 2 0 0 1 20.2 5.04l-.06.06a1.65 1.65 0 0 0-.33 1.82V7c0 .68.39 1.3 1.01 1.58.31.15.64.23.98.23H22a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/></svg>
      )
    case "play":
      return <svg {...common}><polygon points="6 4 20 12 6 20 6 4"/></svg>
    case "pause":
      return <svg {...common}><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
    case "reset":
      return <svg {...common}><path d="M3 12a9 9 0 1 0 3-6.7"/><polyline points="3 3 3 9 9 9"/></svg>
    case "moon":
      return <svg {...common}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    case "sun":
      return <svg {...common}><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
    default:
      return null
  }
}

function Toggle({ checked, onChange, label, tip }: ToggleProps): React.ReactNode {  const id = useMemo(() => Math.random().toString(36).slice(2), [])
  return (
    <label className="switch tooltip" data-tip={tip || ""}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
      <span className="track"><span className="thumb" /></span>
      <span style={{ fontSize: 13, color: "var(--muted)" }}>{label}</span>
    </label>
  )
}

function Progress({ value }: ProgressProps): React.ReactNode {
  return (
    <div className="progress"><span style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></div>
  )
}

/* ---------------------------------- Helpers ---------------------------------- */
function hslBlueRed(t: number): string {
  const hue = 240 - 240 * t
  const sat = 85
  const light = 52 + 18 * t
  return `hsl(${hue} ${sat}% ${light}%)`
}

function downloadFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" })
  const link = document.createElement("a")
  link.href = URL.createObjectURL(blob)
  link.download = filename
  link.click()
  URL.revokeObjectURL(link.href)
}

/* ---------------------------------- Panels ----------------------------------- */

function BrainPanel({ synapses, rewards, params, setParams, metrics }: BrainPanelProps): React.ReactNode {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><Icon name="brain" /> Brain Simulation</div>
        <div className="badges">
          <span className="tag">Neural Activity</span>
          <span className="tag">Plasticity</span>
          <span className="tag">Rewards</span>
        </div>
      </div>
      <div className="panel-body">
        <div className="row">
          <div className="panel pulse">
            <div className="panel-header">
              <div className="panel-title">Synaptic Strength Matrix</div>
              <div className="panel-sub">Hover to inspect</div>
            </div>
            <div className="panel-body">
              <div className="heatmap">
                {synapses.map(s => (
                  <div
                    key={s.id}
                    className="cell"
                    style={{
                      background: `linear-gradient(180deg, ${hslBlueRed(s.strength)} 0%, ${hslBlueRed(Math.min(1, s.strength + 0.2))} 100%)`,
                      boxShadow: s.active ? "0 0 22px rgba(122,162,255,0.35), inset 0 0 8px rgba(39,224,163,0.25)" : "none",
                    }}
                    title={`Synapse #${s.id}\nStrength: ${(s.strength*100).toFixed(1)}%\nHistory: ${s.history.slice(-5).map(v => (v*100).toFixed(0)).join("%, ")}%`}
                    aria-label={`Synapse ${s.id} with strength ${(s.strength*100).toFixed(0)} percent`}
                  />
                ))}
              </div>
              <div className="legend">
                <span>Scale</span>
                <span className="chip" style={{ background: hslBlueRed(0) }} />
                <span className="chip" style={{ background: hslBlueRed(0.25) }} />
                <span className="chip" style={{ background: hslBlueRed(0.5) }} />
                <span className="chip" style={{ background: hslBlueRed(0.75) }} />
                <span className="chip" style={{ background: hslBlueRed(1) }} />
                <span className="panel-sub">Blue (low) → Red (high)</span>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">Brain Summary Metrics</div>
            </div>
            <div className="panel-body">
              <div className="kpi-row">
                <div className="kpi">
                  <h4>Total Reward Mean</h4>
                  <div className="val">{(metrics.rewardMean*100).toFixed(1)}%</div>
                  <Progress value={metrics.rewardMean * 100} />
                </div>
                <div className="kpi">
                  <h4>Reward Std Dev</h4>
                  <div className="val">{(metrics.rewardSd*100).toFixed(1)}%</div>
                  <div className="trend">Stable</div>
                </div>
                <div className="kpi">
                  <h4>Synaptic Avg</h4>
                  <div className="val">{(metrics.synAvg*100).toFixed(1)}%</div>
                  <Progress value={metrics.synAvg * 100} />
                </div>
                <div className="kpi">
                  <h4>State Magnitude</h4>
                  <div className="val">{metrics.stateMag.toFixed(3)}</div>
                  <div className="trend">+{(Math.random()*0.9).toFixed(2)}</div>
                </div>
              </div>

              <div className="hr" />

              <div className="controls">
                <div className="control">
                  <label>Plasticity Learning Rate: {params.plasticity.toFixed(3)}</label>
                  <input className="range" type="range" min="0" max="0.5" step="0.001" value={params.plasticity} onChange={e => setParams(p => ({ ...p, plasticity: parseFloat(e.target.value) }))} />
                </div>
                <div className="control">
                  <label>Noise Injection Level: {params.noise.toFixed(2)}</label>
                  <input className="range" type="range" min="0" max="1" step="0.01" value={params.noise} onChange={e => setParams(p => ({ ...p, noise: parseFloat(e.target.value) }))} />
                </div>
                <Toggle label="Enable Plasticity" checked={params.enablePlasticity} onChange={(v) => setParams(p => ({ ...p, enablePlasticity: v }))} tip="Adaptive synaptic updates during training." />
                <div className="panel-sub">Tip: Toggle plasticity for adaptive learning of the brain model.</div>
              </div>
            </div>
          </div>
        </div>

        <div className="panel" style={{ marginTop: "16px" }}>
          <div className="panel-header">
            <div className="panel-title">Reward Signal Evolution</div>
            <div className="panel-sub">Mean reward per epoch — zoom via mouse wheel + drag</div>
          </div>
          <div className="panel-body" style={{ height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rewards}>
                <defs>
                  <linearGradient id="gradR" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7aa2ff" stopOpacity={0.9}/>
                    <stop offset="100%" stopColor="#a07dff" stopOpacity={0.2}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(158,173,214,0.2)" strokeDasharray="3 3" />
                <XAxis dataKey="epoch" stroke="var(--muted)"/>
                <YAxis domain={[0,1]} tickFormatter={v => `${(v*100)|0}%`} stroke="var(--muted)"/>
                <Tooltip formatter={(v: any, n: any) => n === "reward" ? `${(v*100).toFixed(1)}%` : v} contentStyle={{ background: "#0b1020", border: "1px solid rgba(158,173,214,0.35)", borderRadius: 10, color: "var(--text)" }}/>
                <Area type="monotone" dataKey="reward" stroke="#7aa2ff" strokeWidth={2} fill="url(#gradR)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}


function MoleculesPanel({ molecules, onRegenerate, selected, setSelected, onExport, onFileUpload }: MoleculesPanelProps): React.ReactNode {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><Icon name="flask" /> Molecular Generator</div>
        <div className="panel-sub">Quantum-designed candidates with fingerprint analysis</div>
      </div>
      <div className="panel-body">
        <FileUpload onFileUpload={onFileUpload} onRegenerate={onRegenerate} />
        
        <div className="row">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">Generated Molecules</div>
              <div className="flex gap-8">
                <button className="btn" onClick={onRegenerate}><span>Generate</span></button>
                <button className="btn" onClick={onExport}><span>Export CSV</span></button>
              </div>
            </div>
            <div className="panel-body">
              <div className="mol-list">
                {molecules.map((m, idx) => (
                  <div key={m.id} className={`mol-item ${selected === idx ? "active" : ""}`} onClick={() => setSelected(idx)} role="button" aria-label={`Select molecule ${m.structure}`}>
                    <div className="fp-bits" aria-hidden="true">
                      {m.bits.map((b, i) => <span key={i} className={`bit ${b ? "on" : ""}`} />)}
                    </div>
                    <div>
                      <div style={{ fontWeight: 700 }}>{m.structure}</div>
                      <div className="panel-sub">
                        Similarity: {m.similarity.toFixed(1)}% • Binding: {m.energy.toFixed(2)} kcal/mol • Density: {(m.density*100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="badge">{m.contFp.slice(0, 3).map(v => v.toFixed(2)).join(" • ")}</div>
                  </div>
                ))}
              </div>

              <div className="hr" />

              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={molecules}>
                    <CartesianGrid stroke="rgba(158,173,214,0.2)" strokeDasharray="3 3" />
                    <XAxis dataKey="structure" stroke="var(--muted)"/>
                    <YAxis stroke="var(--muted)"/>
                    <Tooltip contentStyle={{ background: "#0b1020", border: "1px solid rgba(158,173,214,0.35)", borderRadius: 10, color: "var(--text)" }}/>
                    <Bar dataKey="similarity" fill="#27e0a3" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">Molecule Detail</div>
              <div className="panel-sub">Click a molecule to expand details</div>
            </div>
            <div className="panel-body">
              {molecules[selected] ? (
                <>
                  <div className="flex justify-between items-center">
                    <div style={{ fontSize: 18, fontWeight: 800 }}>{molecules[selected].structure}</div>
                    <span className="badge">Tanimoto {molecules[selected].similarity.toFixed(1)}%</span>
                  </div>
                  <div className="hr" />
                  <div className="panel-sub">Binary Fingerprint</div>
                  <div className="heatmap" style={{ gridTemplateColumns: "repeat(8, 1fr)", gap: 4 }}>
                    {molecules[selected].bits.map((b, i) => (
                      <div key={i} className="bit" style={{
                        width: 22, height: 22,
                        background: b ? "linear-gradient(180deg,#3ddc97,#27e0a3)" : "rgba(158,173,214,0.15)",
                        borderColor: b ? "rgba(61,220,151,0.8)" : "rgba(158,173,214,0.25)"
                      }} title={`Bit ${i}: ${b ? "on" : "off"}`} />
                    ))}
                  </div>

                  <div className="hr" />

                  <div className="panel-sub">Continuous Fingerprint (sample)</div>
                  <div className="flex" style={{ gap: 8, flexWrap: "wrap" }}>
                    {molecules[selected].contFp.slice(0, 16).map((v, i) => (
                      <div key={i} className="badge mono" style={{ background: "rgba(18,26,54,0.65)" }}>{v.toFixed(3)}</div>
                    ))}
                  </div>

                  <div className="hr" />
                  <button className="btn" onClick={() => {
                    const smiles = molecules[selected]?.smiles || ''
                    const elementId = 'viewer3d'
                    let container = document.getElementById(elementId)
                    if (!container) {
                      container = document.createElement('div')
                      container.id = elementId
                      container.style.width = '100%'
                      container.style.height = '360px'
                      container.style.marginTop = '10px'
                      container.style.border = '1px solid rgba(158,173,214,0.25)'
                      container.style.borderRadius = '10px'
                      const panel = document.querySelector('.container')
                      panel && panel.appendChild(container)
                    }
                    if (!(window as any).$3Dmol) { alert('3Dmol.js not loaded'); return }
                    const viewer = (window as any).$3Dmol.createViewer(elementId, { backgroundColor: 'white' })
                    viewer.clear()
                    const fallback = 'Cn1cnc2n(C)c(=O)n(C)c(=O)c12' // caffeine as safe default
                    const smi = smiles && smiles.trim().length > 0 ? smiles : fallback
                    try {
                      const model = viewer.addModel(smi, 'smiles')
                      viewer.setStyle({}, { stick: { radius: 0.18 }, sphere: { scale: 0.22 } })
                      viewer.zoomTo()
                      viewer.render()
                    } catch (e) {
                      alert('Could not render this SMILES. Showing fallback (caffeine).')
                      viewer.addModel(fallback, 'smiles')
                      viewer.setStyle({}, { stick: { radius: 0.18 }, sphere: { scale: 0.22 } })
                      viewer.zoomTo()
                      viewer.render()
                    }
                  }}>Preview Structure</button>
                </>
              ) : (
                <div className="panel-sub">No molecule selected.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function QuantumPanel({ binding, vqeSteps, setVqeSteps, onRerun }: QuantumPanelProps): React.ReactNode {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><Icon name="atom" /> Quantum Binding (VQE)</div>
        <div className="panel-sub">Binding energy optimization and convergence</div>
      </div>
      <div className="panel-body">
        <div className="row">
          <div className="panel">
            <div className="panel-header"><div className="panel-title">Binding Energy Evolution</div></div>
            <div className="panel-body" style={{ height: 320 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={binding}>
                  <CartesianGrid stroke="rgba(158,173,214,0.2)" strokeDasharray="3 3" />
                  <XAxis dataKey="step" stroke="var(--muted)"/>
                  <YAxis stroke="var(--muted)"/>
                  <Tooltip contentStyle={{ background: "#0b1020", border: "1px solid rgba(158,173,214,0.35)", borderRadius: 10, color: "var(--text)" }}/>
                  <Line type="monotone" dataKey="energy" stroke="#a07dff" strokeWidth={2} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header"><div className="panel-title">Current Binding Energy</div></div>
            <div className="panel-body">
              <div className="kpi-row" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
                <div className="kpi">
                  <h4>Latest Energy</h4>
                  <div className="val">{binding[binding.length - 1]?.energy.toFixed(3)}</div>
                  <div className="panel-sub">kcal/mol</div>
                </div>
                <div className="kpi">
                  <h4>Convergence</h4>
                  <div className="val">{binding[binding.length - 1]?.convergence.toFixed(0)}%</div>
                  <Progress value={binding[binding.length - 1]?.convergence || 0}/>
                </div>
              </div>

              <div className="hr" />

              <div className="control">
                <label>Max VQE Optimization Steps: {vqeSteps}</label>
                <input className="range" type="range" min="10" max="150" step="5" value={vqeSteps} onChange={e => setVqeSteps(parseInt(e.target.value, 10))}/>
              </div>
              <div className="flex gap-10" style={{ marginTop: 10 }}>
                <button className="btn" onClick={onRerun}><span>Re-run VQE</span></button>
                <button className="btn" onClick={() => alert("Add noise model controls here (depolarizing, amplitude damping, readout).")}>Noise / Error Injection</button>
              </div>

              <div className="hr" />
              <div className="panel-sub" style={{ marginBottom: 8 }}>Hamiltonian (example)</div>
              <div className="mono scroll" style={{ border: "1px solid rgba(158,173,214,0.25)", borderRadius: 10, padding: 10, background: "rgba(18,26,54,0.65)" }}>
{`H = -2.103 * Z₀Z₁ + 1.742 * X₀X₁
  + 0.803 * Y₀Y₁ - 1.212 * Z₀
  - 0.914 * Z₁ + 3.402 * I
  + λ * (fingerprint-weighted projector terms)
`}
              </div>
            </div>
          </div>
        </div>

        <div className="panel" style={{ marginTop: 16 }}>
          <div className="panel-header"><div className="panel-title">Convergence Analysis</div></div>
          <div className="panel-body" style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={binding}>
                <defs>
                  <linearGradient id="gradC" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ffb74a" stopOpacity={0.9}/>
                    <stop offset="100%" stopColor="#ffb74a" stopOpacity={0.15}/>
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(158,173,214,0.2)" strokeDasharray="3 3" />
                <XAxis dataKey="step" stroke="var(--muted)"/>
                <YAxis stroke="var(--muted)"/>
                <Tooltip contentStyle={{ background: "#0b1020", border: "1px solid rgba(158,173,214,0.35)", borderRadius: 10, color: "var(--text)" }}/>
                <Area type="monotone" dataKey="convergence" stroke="#ffb74a" fill="url(#gradC)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

function ControlsPanel({ status, setStatus, params, setParams, onStartPause, onReset, onSave, onLoad }: ControlsPanelProps): React.ReactNode {
  const fileRef = useRef<HTMLInputElement>(null)
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><Icon name="settings" /> System Controls</div>
        <div className="panel-sub">Configure training, modules, and monitor status</div>
      </div>
      <div className="panel-body">
        <div className="row">
          <div className="panel">
            <div className="panel-header"><div className="panel-title">Training Parameters</div></div>
            <div className="panel-body">
              <div className="controls">
                <div className="control">
                  <label>Epochs ({params.epochs})</label>
                  <input className="range" type="range" min="10" max="2000" step="10" value={params.epochs} onChange={e => setParams(p => ({ ...p, epochs: parseInt(e.target.value, 10) }))}/>
                </div>
                <div className="control">
                  <label>Batch Size</label>
                  <input className="input" type="number" min="1" value={params.batch} onChange={e => setParams(p => ({ ...p, batch: parseInt(e.target.value || "1", 10) }))}/>
                </div>
                <div className="control">
                  <label>Learning Rates</label>
                  <div className="flex gap-10">
                    <input className="input" style={{ maxWidth: 180 }} type="number" step="0.0001" value={params.lrGen} onChange={e => setParams(p => ({ ...p, lrGen: parseFloat(e.target.value || "0") }))} placeholder="Quantum Generator"/>
                    <input className="input" style={{ maxWidth: 180 }} type="number" step="0.0001" value={params.lrDisc} onChange={e => setParams(p => ({ ...p, lrDisc: parseFloat(e.target.value || "0") }))} placeholder="Discriminator"/>
                  </div>
                </div>
                <div className="control">
                  <label>VQE Penalty λ: {params.lambda.toFixed(2)}</label>
                  <input className="range" type="range" min="0" max="2" step="0.01" value={params.lambda} onChange={e => setParams(p => ({ ...p, lambda: parseFloat(e.target.value) }))}/>
                </div>
                <div className="control">
                  <label>Random Seed</label>
                  <input className="input" type="number" value={params.seed} onChange={e => setParams(p => ({ ...p, seed: parseInt(e.target.value || "0", 10) }))}/>
                </div>
                <div className="flex gap-10">
                  <Toggle label="Brain Feedback Loop" checked={params.modBrain} onChange={v => setParams(p => ({ ...p, modBrain: v }))} />
                  <Toggle label="VQE Penalty" checked={params.modVqe} onChange={v => setParams(p => ({ ...p, modVqe: v }))} />
                  <Toggle label="Plasticity" checked={params.modPlas} onChange={v => setParams(p => ({ ...p, modPlas: v }))} />
                  <Toggle label="GAN Discriminator" checked={params.modDisc} onChange={v => setParams(p => ({ ...p, modDisc: v }))} />
                </div>
              </div>
              <div className="hr" />
              <div className="flex gap-10">
                <button className="btn primary" onClick={onStartPause}>
                  <Icon name={status.running ? "pause" : "play"} />
                  <span>{status.running ? "Pause" : "Run"}</span>
                </button>
                <button className="btn" onClick={onReset}><Icon name="reset" /><span>Reset</span></button>
                <button className="btn" onClick={onSave}>Save Config</button>
                <button className="btn" onClick={() => fileRef.current?.click()}>Load Config</button>
                <input ref={fileRef} type="file" accept="application/json" hidden onChange={onLoad}/>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header"><div className="panel-title">Real-Time System Status</div></div>
            <div className="panel-body">
              <div className="kpi-row">
                <div className="kpi">
                  <h4>Epoch</h4>
                  <div className="val">{status.epoch}</div>
                </div>
                <div className="kpi">
                  <h4>GAN Loss</h4>
                  <div className="val">{status.ganLoss.toFixed(3)}</div>
                </div>
                <div className="kpi">
                  <h4>Disc Loss</h4>
                  <div className="val">{status.discLoss.toFixed(3)}</div>
                </div>
                <div className="kpi">
                  <h4>VQE Penalty</h4>
                  <div className="val">{status.vqePenalty.toFixed(3)}</div>
                </div>
              </div>
              <div style={{ marginTop: 10 }}>
                <label className="panel-sub">Training Progress</label>
                <Progress value={status.progress} />
              </div>

              <div className="hr" />
              <div className="panel-sub" style={{ marginBottom: 8 }}>Live Logs</div>
              <div className="mono scroll">
                {status.logs.map((l, i) => (
                  <div key={i} style={{ color: l.level === "warn" ? "var(--warn)" : l.level === "err" ? "var(--danger)" : "var(--muted)" }}>
                    [{l.level.toUpperCase()}] {l.msg}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------- Authentication Components ------------------------------- */

function LoginSignup({ onLogin, onSignup, isLogin = true }: LoginSignupProps): React.ReactNode {
  const [email, setEmail] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault()
    setLoading(true)
    setError('')
    
    // Simulate loading delay
    setTimeout(() => {
      // Always succeed for demo purposes
      const demoToken = 'demo_token_' + Date.now()
      localStorage.setItem('token', demoToken)
      
      if (isLogin) {
        onLogin(demoToken, email)
      } else {
        onSignup(demoToken, email)
      }
      setLoading(false)
    }, 1000)
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>{isLogin ? 'Login' : 'Sign Up'}</h2>
          <p>Access your QBrainX workspace</p>
        </div>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="Enter your password"
            />
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <button type="submit" className="auth-btn" disabled={loading}>
            {loading ? 'Processing...' : (isLogin ? 'Login' : 'Sign Up')}
          </button>
        </form>
      </div>
    </div>
  )
}

function UserHistory({ history, onLoadHistory }: UserHistoryProps): React.ReactNode {
  return (
    <div className="history-panel">
      <div className="panel-header">
        <div className="panel-title">Run History</div>
        <div className="panel-sub">Your previous simulation results</div>
      </div>
      <div className="panel-body">
        {history.length === 0 ? (
          <div className="panel-sub">No previous runs found. Start training to see your history here.</div>
        ) : (
          <div className="history-list">
            {history.map((entry, index) => (
              <div key={index} className="history-item" onClick={() => onLoadHistory(entry)}>
                <div className="history-date">
                  {new Date(entry.ts).toLocaleString()}
                </div>
                <div className="history-summary">
                  Reward: {(entry.summary?.rewardMean * 100 || 0).toFixed(1)}% • 
                  Epochs: {entry.summary?.epochs || 0} • 
                  GAN Loss: {(entry.summary?.ganLoss || 0).toFixed(3)} • 
                  VQE Penalty: {(entry.summary?.vqePenalty || 0).toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function FileUpload({ onFileUpload, onRegenerate }: FileUploadProps): React.ReactNode {
  const [uploading, setUploading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError('')

    try {
      const content = await file.text()
      
      // Parse content based on file type
      let smilesList: string[] = [];

    if (file.name.endsWith('.csv')) {
      // Handle CSV format
      const lines = content.split('\n').filter(line => line.trim());
      smilesList = lines
        .map(line => line.split(',')[0]?.trim()) // safely access first column
        .filter(smiles => smiles && !smiles.startsWith('#')) as string[]; // ensure type
    } else {
      // Handle TXT format
      smilesList = content
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'));
    }


      if (smilesList.length === 0) {
        throw new Error('No valid SMILES strings found in file')
      }

      // Store in localStorage for demo purposes
      localStorage.setItem('uploaded_smiles', JSON.stringify(smilesList))
      
      onFileUpload(`Successfully processed ${smilesList.length} SMILES strings`)
    } catch (err) {
      setError((err as Error).message || 'Failed to process file')
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  return (
    <div className="file-upload-section">
      <div className="upload-controls">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.txt"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        <button 
          className="btn" 
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? 'Processing...' : 'Upload SMILES File'}
        </button>
        <button className="btn" onClick={onRegenerate}>
          Generate Default Molecules
        </button>
      </div>
      {error && <div className="error-message">{error}</div>}
      <div className="panel-sub">
        Upload a CSV or TXT file with SMILES strings (one per line) to analyze custom molecules.
      </div>
    </div>
  )
}

/* ------------------------------------- App ------------------------------------ */

export default function App(): React.ReactNode {
  const [dark, setDark] = useState<boolean>(true)
  const [tab, setTab] = useState<string>("overview")
  
  // Authentication state
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [showLogin, setShowLogin] = useState<boolean>(true)
  const [userHistory, setUserHistory] = useState<HistoryEntry[]>([])

  // Brain state
  const [synapses, setSynapses] = useState<Synapse[]>(() => genSynapses())
  const [rewards, setRewards] = useState<RewardData[]>(() => genRewards())
  const [binding, setBinding] = useState<BindingData[]>(() => genBinding())
  const [molecules, setMolecules] = useState<Molecule[]>(() => genMolecules())
  const [selectedMol, setSelectedMol] = useState<number>(0)

  const [vqeSteps, setVqeSteps] = useState<number>(60)

  const [params, setParams] = useState<SystemParams>({
    plasticity: 0.12,
    noise: 0.18,
    enablePlasticity: true,
    epochs: 350,
    batch: 32,
    lrGen: 0.0008,
    lrDisc: 0.0012,
    lambda: 0.55,
    seed: 42,
    modBrain: true, 
    modVqe: true, 
    modPlas: true, 
    modDisc: true,
  })

  const [status, setStatus] = useState<SystemStatus>({
    running: false,
    epoch: 0,
    progress: 0,
    ganLoss: 0.46,
    discLoss: 0.39,
    vqePenalty: 0.14,
    logs: [
      { level: "info", msg: "QBrainX initialized." },
      { level: "info", msg: "Quantum circuits prepared." },
      { level: "info", msg: "Awaiting user start..." },
    ],
  })

  // Authentication handlers
  const handleLogin = (newToken: string, email: string): void => {
    setToken(newToken)
    setUser({ email })
    setShowLogin(true)
    loadUserHistory(newToken)
  }

  const handleSignup = (newToken: string, email: string): void => {
    setToken(newToken)
    setUser({ email })
    setShowLogin(true)
    setUserHistory([])
  }

  const handleLogout = (): void => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
    setUserHistory([])
  }

  const loadUserHistory = async (userToken: string): Promise<void> => {
    try {
      // For demo purposes, load from localStorage or create empty history
      const storedHistory = localStorage.getItem('user_history')
      if (storedHistory) {
        setUserHistory(JSON.parse(storedHistory))
      } else {
        setUserHistory([])
      }
    } catch (error) {
      console.error('Failed to load history:', error)
      setUserHistory([])
    }
  }

  const saveUserHistory = (entry: HistoryEntry): void => {
    try {
      const newHistory = [...userHistory, entry]
      setUserHistory(newHistory)
      localStorage.setItem('user_history', JSON.stringify(newHistory))
    } catch (error) {
      console.error('Failed to save history:', error)
    }
  }

  // Load user history on mount if token exists
  useEffect(() => {
    if (token) {
      loadUserHistory(token)
    }
  }, [token])

  const metrics = useMemo<Metrics>(() => {
    const rewardMean = rewards.slice(-30).reduce((a, b) => a + b.reward, 0) / Math.min(30, rewards.length)
    const rewardSd = Math.sqrt(rewards.slice(-30).reduce((a, b) => a + Math.pow(b.reward - rewardMean, 2), 0) / Math.min(30, rewards.length))
    const synAvg = synapses.reduce((a, b) => a + b.strength, 0) / synapses.length
    const stateMag = Math.sqrt(synapses.reduce((a, b) => a + b.strength * b.strength, 0))
    return { rewardMean, rewardSd, synAvg, stateMag }
  }, [rewards, synapses])

  // Training simulation loop
  useEffect(() => {
    if (!status.running) return
    const t = setInterval(() => {
      setStatus(s => {
        const nextEpoch = s.epoch + 1
        const nextProgress = Math.min(100, s.progress + Math.random() * 1.8)
        const nextGan = Math.max(0.08, s.ganLoss - 0.0025 + (Math.random() - 0.5) * 0.003)
        const nextDisc = Math.max(0.08, s.discLoss - 0.002 + (Math.random() - 0.5) * 0.003)
        const nextVqe = Math.max(0.03, s.vqePenalty - 0.0012 + (Math.random() - 0.5) * 0.002)

        // Save history entry when training completes
        if (nextProgress >= 100 && s.progress < 100) {
          const rewardMean = metrics.rewardMean || 0
          saveUserHistory({
            ts: Date.now(),
            summary: { 
              rewardMean, 
              bindingSteps: binding.length,
              epochs: nextEpoch,
              ganLoss: nextGan,
              discLoss: nextDisc,
              vqePenalty: nextVqe
            }
          })
        }

        return {
          ...s,
          epoch: nextEpoch,
          progress: nextProgress,
          ganLoss: nextGan,
          discLoss: nextDisc,
          vqePenalty: nextVqe,
          logs: [
            ...s.logs.slice(-220),
            { level: "debug", msg: `Epoch ${nextEpoch}: reward ${(Math.random()*100).toFixed(1)}%` },
            { level: "info", msg: `Loss(G,D,VQE) = ${nextGan.toFixed(3)}, ${nextDisc.toFixed(3)}, ${nextVqe.toFixed(3)}` },
          ],
        }
      })
      setSynapses(genSynapses())
      setRewards(prev => {
        const lastEpoch = prev.length ? (prev[prev.length - 1]?.epoch ?? 0) + 1 : 0;
        return [...prev.slice(-200), ...genRewards(1, lastEpoch)]
      })
      setBinding(prev => {
        const last = prev[prev.length - 1] || { step: 0, energy: -32, convergence: 0 }
        const next = { step: last.step + 1, energy: last.energy - Math.exp(-last.step * 0.05) * 0.8 + (Math.random() - 0.5) * 0.6, convergence: Math.min(100, last.convergence + Math.random() * 2) }
        return [...prev.slice(-120), next]
      })
    }, 850)
    return () => clearInterval(t)
  }, [status.running, metrics.rewardMean, binding.length, saveUserHistory])

  useEffect(() => {
    if (!token) return
    
    const ws = new WebSocket('ws://localhost:3003')

    ws.onopen = () => {
      console.log('Connected to backend WebSocket')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'synapses') setSynapses(data.payload)
      else if (data.type === 'rewards') setRewards(data.payload)
      else if (data.type === 'binding') setBinding(data.payload)
      else if (data.type === 'molecules') setMolecules(data.payload)
      else if (data.type === 'status') setStatus(data.payload)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
    }

    return () => ws.close()
  }, [token])

  // Theme toggle
  useEffect(() => {
    document.documentElement.style.colorScheme = dark ? "dark" : "light"
  }, [dark])

  function handleExportMolecules(): void {
    const header = ["id", "structure", "similarity", "bindingEnergy", "bitDensity", "bits"].join(",")
    const lines = molecules.map(m =>
      [m.id, m.structure, m.similarity.toFixed(4), m.energy.toFixed(4), m.density.toFixed(4), `"${m.bits.join("")}"`].join(",")
    )
    downloadFile("qbrainx_molecules.csv", [header, ...lines].join("\n"))
  }

  function handleSaveConfig(): void {
    const cfg = { params, vqeSteps, tab, seed: params.seed }
    downloadFile("qbrainx_config.json", JSON.stringify(cfg, null, 2))
  }

  function handleLoadConfig(e: React.ChangeEvent<HTMLInputElement>): void {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const cfg = JSON.parse(reader.result as string)
        if (cfg.params) setParams(cfg.params)
        if (cfg.vqeSteps) setVqeSteps(cfg.vqeSteps)
        if (cfg.tab) setTab(cfg.tab)
        alert("Configuration loaded.")
      } catch {
        alert("Invalid configuration file.")
      }
    }
    reader.readAsText(file)
  }

  function rerunVQE(): void {
    setBinding(genBinding(vqeSteps))
    setStatus(s => ({ ...s, logs: [...s.logs, { level: "info", msg: `VQE re-run with ${vqeSteps} steps.` }] }))
  }

  const tabs: Tab[] = [
    { id: "overview", label: "Overview", icon: "brain" },
    { id: "brain", label: "Brain Simulation", icon: "brain" },
    { id: "molecules", label: "Molecular Generator", icon: "flask" },
    { id: "quantum", label: "Quantum Binding", icon: "atom" },
    { id: "controls", label: "System Controls", icon: "settings" },
  ]

  // Show login/signup if not authenticated
  if (!token) {
    return (
      <div className="container" role="main">
        <header className="header" role="banner">
          <div className="header-row">
            <div className="brand">
              <div className="logo" aria-hidden="true">
                <Icon name="brain" size={22} />
              </div>
              <div className="title">
                <h1>QBrainX</h1>
                <small>Quantum-Accelerated Neuroscience & Drug Discovery</small>
              </div>
            </div>
          </div>
        </header>
        
        <LoginSignup 
          onLogin={handleLogin}
          onSignup={handleSignup}
          isLogin={showLogin}
        />
        
        <div style={{ textAlign: 'center', marginTop: '20px' }}>
          <button 
            className="btn" 
            onClick={() => setShowLogin(!showLogin)}
            style={{ background: 'transparent', border: '1px solid var(--muted)' }}
          >
            {showLogin ? 'Need an account? Sign up' : 'Have an account? Login'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="container" role="main">
      <header className="header" role="banner">
        <div className="header-row">
          <div className="brand">
            <div className="logo" aria-hidden="true">
              <Icon name="brain" size={22} />
            </div>
            <div className="title">
              <h1>QBrainX</h1>
              <small>Quantum-Accelerated Neuroscience & Drug Discovery</small>
            </div>
          </div>
          <div className="header-actions">
            <span className="badge">
              <span className="tooltip" data-tip={status.running ? "Training running" : "Training paused"}>
                {status.running ? "Active" : "Idle"}
              </span>
            </span>
            <span className="badge" style={{ background: 'rgba(39,224,163,0.2)', color: '#27e0a3' }}>
              {user?.email}
            </span>
            <button className="btn" onClick={handleLogout}>
              Logout
            </button>
            <button className="btn" onClick={() => setDark(d => !d)} aria-label="Toggle theme">
              <Icon name={dark ? "sun" : "moon"} />
              <span>{dark ? "Light" : "Dark"}</span>
            </button>
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Primary">
        {tabs.map(t => (
          <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)} aria-pressed={tab === t.id}>
            <Icon name={t.icon} />
            <span>{t.label}</span>
          </button>
        ))}
      </nav>

      {tab === "overview" && (
        <>
          <div className="kpi-row">
            <div className="kpi">
              <h4>Current Epoch</h4>
              <div className="val">{status.epoch}</div>
              <Progress value={status.progress} />
            </div>
            <div className="kpi">
              <h4>Brain Reward</h4>
              <div className="val">{(metrics.rewardMean * 100).toFixed(1)}%</div>
              <div className="trend">+{(Math.random()*3).toFixed(1)}% vs last</div>
            </div>
            <div className="kpi">
              <h4>GAN Loss</h4>
              <div className="val">{status.ganLoss.toFixed(3)}</div>
              <div className="panel-sub">Converging</div>
            </div>
            <div className="kpi">
              <h4>VQE Energy</h4>
              <div className="val">{binding[binding.length - 1]?.energy.toFixed(2)}</div>
              <div className="panel-sub">kcal/mol</div>
            </div>
          </div>

          <div className="grid">
            <div className="panel pulse">
              <div className="panel-header">
                <div className="panel-title">System Performance</div>
                <div className="panel-sub">Recent reward trajectory</div>
              </div>
              <div className="panel-body" style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rewards.slice(-30)}>
                    <CartesianGrid stroke="rgba(158,173,214,0.2)" strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" stroke="var(--muted)" />
                    <YAxis domain={[0,1]} tickFormatter={v => `${(v*100)|0}%`} stroke="var(--muted)" />
                    <Tooltip contentStyle={{ background: "#0b1020", border: "1px solid rgba(158,173,214,0.35)", borderRadius: 10, color: "var(--text)" }}/>
                    <Line type="monotone" dataKey="reward" stroke="#7aa2ff" strokeWidth={2} dot={false}/>
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <div className="panel-title">Neural Activity Heatmap</div>
              </div>
              <div className="panel-body">
                <div className="heatmap">
                  {synapses.map(s => (
                    <div key={s.id} className="cell" style={{ background: hslBlueRed(s.strength) }} title={`Strength ${(s.strength*100).toFixed(1)}%`} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <UserHistory 
            history={userHistory} 
            onLoadHistory={(entry) => {
              // Load historical data (simplified - in real app would restore full state)
              setStatus(s => ({ ...s, logs: [...s.logs, { level: "info", msg: `Loaded run from ${new Date(entry.ts).toLocaleString()}` }] }))
            }} 
          />
        </>
      )}

      
      {tab === "brain" && (
        <BrainPanel
          synapses={synapses}
          rewards={rewards}
          params={{
            plasticity: params.plasticity,
            noise: params.noise,
            enablePlasticity: params.enablePlasticity,
          }}
          setParams={fn =>
            setParams(prev => {
              const brainParams = {
                plasticity: prev.plasticity,
                noise: prev.noise,
                enablePlasticity: prev.enablePlasticity,
              }
              const updated = fn(brainParams)
              return { ...prev, ...updated }
            })
          }
          metrics={metrics}
        />
      )}

      {tab === "molecules" && (
        <MoleculesPanel
          molecules={molecules}
          selected={selectedMol}
          setSelected={setSelectedMol}
          onRegenerate={() => setMolecules(genMolecules(10))}
          onExport={handleExportMolecules}
          onFileUpload={(message) => {
            setMolecules(genMolecules(10)); // Simulate new molecules from uploaded file
            setStatus(s => ({ ...s, logs: [...s.logs, { level: "info", msg: `Molecules loaded from file: ${message}` }] }));
          }}
        />
      )}

      {tab === "quantum" && (
        <QuantumPanel
          binding={binding}
          vqeSteps={vqeSteps}
          setVqeSteps={setVqeSteps}
          onRerun={rerunVQE}
        />
      )}

      {tab === "controls" && (
        <ControlsPanel
          status={status}
          setStatus={setStatus}
          params={params}
          setParams={setParams}
          onStartPause={() => setStatus(s => ({ ...s, running: !s.running, logs: [...s.logs, { level: "info", msg: s.running ? "Paused training." : "Started training." }] }))}
          onReset={() => {
            setStatus(s => ({ ...s, running: false, epoch: 0, progress: 0, ganLoss: 0.46, discLoss: 0.39, vqePenalty: 0.14, logs: [...s.logs, { level: "warn", msg: "System reset." }] }))
            setSynapses(genSynapses())
            setRewards(genRewards())
            setBinding(genBinding())
          }}
          onSave={handleSaveConfig}
          onLoad={handleLoadConfig}
        />
      )}

      <footer style={{ marginTop: 26, opacity: 0.8, color: "var(--muted)", fontSize: 12 }}>
        © {new Date().getFullYear()} QBrainX · Ethan Poon Quantum-Classical Hybrid UI Demo.
      </footer>
    </div>
  )
}
