import { useState, useRef } from 'react'
import { Upload, ImageIcon, AlertTriangle, CheckCircle2, Loader2, X } from 'lucide-react'
import './App.css'

interface ClassStat {
  pixel_count: number
  percentage: number
}

interface DamageReport {
  tile_id: string
  generated_at: string
  model_version: string
  summary: {
    total_pixels_analyzed: number
    affected_area_percentage: number
  }
  damage_distribution: Record<string, ClassStat>
  confidence: {
    overall_mean: number
    per_class: Record<string, number | null>
  }
  quality_flags: {
    requires_human_review: boolean
    low_confidence_classes: string[]
  }
  limitations: string[]
}

const API_BASE = 'http://127.0.0.1:8000'

const DAMAGE_COLORS: Record<string, string> = {
  background: 'bg-slate-300',
  'no-damage': 'bg-emerald-500',
  'minor-damage': 'bg-yellow-400',
  'major-damage': 'bg-orange-500',
  destroyed: 'bg-red-600',
}

const DAMAGE_LABELS: Record<string, string> = {
  background: 'Background',
  'no-damage': 'No Damage',
  'minor-damage': 'Minor Damage',
  'major-damage': 'Major Damage',
  destroyed: 'Destroyed',
}

function ImageDropzone({
  label,
  file,
  onChange,
}: {
  label: string
  file: File | null
  onChange: (f: File | null) => void
}) {
  const [isDragging, setIsDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File | null) => {
    onChange(f)
    if (f) {
      setPreviewUrl(URL.createObjectURL(f))
    } else {
      setPreviewUrl(null)
    }
  }

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">{label}</label>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          const f = e.dataTransfer.files?.[0] ?? null
          handleFile(f)
        }}
        onClick={() => inputRef.current?.click()}
        className={`relative rounded-xl border-2 border-dashed transition-colors cursor-pointer overflow-hidden
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100'}
          ${file ? 'h-40' : 'h-40 flex flex-col items-center justify-center'}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {file && previewUrl ? (
          <>
            <img src={previewUrl} alt={`${label} preview`} className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-black/40 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
              <span className="text-white text-sm font-medium">Click to change</span>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                handleFile(null)
                if (inputRef.current) inputRef.current.value = ''
              }}
              className="absolute top-2 right-2 bg-white/90 hover:bg-white rounded-full p-1 shadow"
              aria-label={`Remove ${label}`}
            >
              <X size={14} className="text-slate-700" />
            </button>
          </>
        ) : (
          <>
            <Upload size={28} className={isDragging ? 'text-blue-500' : 'text-slate-400'} />
            <p className="text-sm text-slate-500 mt-2">Drag & drop or click to upload</p>
            <p className="text-xs text-slate-400 mt-1">PNG, JPG</p>
          </>
        )}
      </div>
      {file && (
        <p className="text-xs text-slate-600 mt-1.5 flex items-center gap-1">
          <ImageIcon size={12} /> {file.name}
        </p>
      )}
    </div>
  )
}

function App() {
  const [preFile, setPreFile] = useState<File | null>(null)
  const [postFile, setPostFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState<DamageReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  const canAnalyze = preFile !== null && postFile !== null && !loading

  const handleAnalyze = async () => {
    if (!preFile || !postFile) {
      setError('Please select both a pre-disaster and post-disaster image.')
      return
    }
    setLoading(true)
    setError(null)
    setReport(null)

    const formData = new FormData()
    formData.append('pre_image', preFile)
    formData.append('post_image', postFile)

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(detail?.detail || `Request failed: ${res.status}`)
      }
      const data: DamageReport = await res.json()
      setReport(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Is the API server running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-slate-900 text-white border-b border-gray-800">
        <div className="max-w-4xl mx-auto px-6 py-3 flex items-center min-h-[56px]">
          <h1 className="text-xl font-bold">Intelligent Disaster Response & Damage Assessment</h1>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-6">
          <h2 className="font-semibold text-slate-800 mb-4 text-base">1. Upload Imagery</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">
            <ImageDropzone label="Pre-disaster image" file={preFile} onChange={setPreFile} />
            <ImageDropzone label="Post-disaster image" file={postFile} onChange={setPostFile} />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!canAnalyze}
            className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 font-medium px-5 py-2.5 rounded-lg transition-colors
              ${canAnalyze
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-slate-200 text-slate-400 cursor-not-allowed'}`}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Analyzing...
              </>
            ) : (
              'Run Damage Assessment'
            )}
          </button>

          {error && (
            <div className="mt-4 flex items-start gap-2 bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-3 py-2">
              <AlertTriangle size={16} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </section>

        {report && (
          <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex flex-wrap justify-between items-start gap-3 mb-5 pb-5 border-b border-slate-100">
              <div>
                <h2 className="font-semibold text-slate-800 text-base">2. Assessment Results</h2>
                <p className="text-xs text-slate-500 mt-1">
                  {report.tile_id} · Model {report.model_version} ·{' '}
                  {new Date(report.generated_at).toLocaleString()}
                </p>
              </div>
              {report.quality_flags.requires_human_review ? (
                <span className="inline-flex items-center gap-1.5 bg-amber-100 text-amber-900 text-xs font-semibold px-3 py-1.5 rounded-full">
                  <AlertTriangle size={13} /> Requires Human Review
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 bg-emerald-100 text-emerald-800 text-xs font-semibold px-3 py-1.5 rounded-full">
                  <CheckCircle2 size={13} /> High Confidence
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div className="bg-slate-50 rounded-xl p-4">
                <p className="text-xs text-slate-500 mb-1">Affected Area</p>
                <p className="text-2xl font-bold text-slate-800">{report.summary.affected_area_percentage}%</p>
              </div>
              <div className="bg-slate-50 rounded-xl p-4">
                <p className="text-xs text-slate-500 mb-1">Overall Confidence</p>
                <p className="text-2xl font-bold text-slate-800">
                  {(report.confidence.overall_mean * 100).toFixed(0)}%
                </p>
              </div>
              <div className="bg-slate-50 rounded-xl p-4">
                <p className="text-xs text-slate-500 mb-1">Pixels Analyzed</p>
                <p className="text-2xl font-bold text-slate-800">
                  {report.summary.total_pixels_analyzed.toLocaleString()}
                </p>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-slate-700 mb-3">Damage Distribution</h3>
            <div className="space-y-2.5 mb-6">
              {Object.entries(report.damage_distribution).map(([cls, stat]) => (
                <div key={cls} className="flex items-center gap-3">
                  <span className="text-sm text-slate-600 w-28 shrink-0">{DAMAGE_LABELS[cls] ?? cls}</span>
                  <div className="flex-1 bg-slate-100 rounded-full h-2.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${DAMAGE_COLORS[cls] ?? 'bg-slate-300'}`}
                      style={{ width: `${Math.min(stat.percentage, 100)}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-slate-700 w-16 text-right shrink-0">
                    {stat.percentage === 0 && stat.pixel_count > 0 ? '<0.01' : stat.percentage}%
                  </span>
                </div>
              ))}
            </div>

            <h3 className="text-sm font-semibold text-slate-700 mb-3">Per-Class Confidence</h3>
            <div className="grid grid-cols-5 gap-2 mb-6">
              {Object.entries(report.confidence.per_class).map(([cls, conf]) => {
                const isLow = conf !== null && conf < 0.5
                return (
                  <div
                    key={cls}
                    className={`text-center rounded-lg py-2.5 border ${
                      isLow ? 'bg-red-50 border-red-200' : 'bg-slate-50 border-slate-200'
                    }`}
                  >
                    <div className="text-[11px] text-slate-500 mb-0.5 truncate px-1">
                      {DAMAGE_LABELS[cls] ?? cls}
                    </div>
                    <div className={`text-sm font-bold ${isLow ? 'text-red-700' : 'text-slate-700'}`}>
                      {conf !== null ? conf.toFixed(2) : 'n/a'}
                    </div>
                  </div>
                )
              })}
            </div>

            <details className="text-sm">
              <summary className="cursor-pointer font-semibold text-slate-700 mb-2">
                Limitations & Disclaimers
              </summary>
              <ul className="text-xs text-slate-600 list-disc list-inside space-y-1.5 mt-2 bg-slate-50 rounded-lg p-4">
                {report.limitations.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            </details>
          </section>
        )}
      </main>
    </div>
  )
}

export default App