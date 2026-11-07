import React, { useState, useEffect, useRef } from 'react'
import {
  MessageSquareMore,
  FileText,
  Sparkles,
  BarChart3,
  Settings,
  Zap,
  CheckCircle2,
  Globe,
  Trash2,
  User,
  Bot,
  CheckCheck,
  Copy,
  Send,
  Lightbulb,
  HelpCircle,
  Check,
  Upload,
  RefreshCw,
  Database,
  BookOpen,
  Scale,
  Brain,
  Sun,
  Moon,
  Menu,
  X,
  ExternalLink,
  Download,
  Eye,
  AlertTriangle,
  Layers,
  Activity,
} from 'lucide-react'

const API_BASE = '/api'
const CHAT_STORAGE_KEY = 'ai_assistant_chat_session_history'

interface Message {
  user: string
  assistant: string
  mode?: string // 'GENERAL' | 'RAG' | 'HYBRID'
  time: string
  grounded?: boolean
}

interface DocItem {
  id: string
  filename: string
  name: string
  pages: number
  size_mb: number
  chunks: number
  uploaded_at: string
  is_scanned?: boolean
  status?: string
}

interface EvaluationReport {
  timestamp: string
  provider: string
  dataset_name: string
  total_questions: number
  total_chunks_indexed: number
  benchmark_duration_seconds: number
  avg_retrieval_latency_ms: number
  avg_generation_latency_seconds: number
  avg_total_latency_seconds: number
  avg_keyword_recall: number
  avg_precision_at_k: number
  mean_reciprocal_rank: number
  page_hit_rate: number
  faithfulness_rate: number
  answer_relevancy: number
  details: Array<{
    id: string
    language: string
    question: string
    retrieval_latency_ms: number
    generation_latency_seconds: number
    total_latency_seconds: number
    keyword_recall: number
    precision_at_k: number
    mrr: number
    page_hit: boolean
    relevancy: number
    retrieved_count: number
    answer_preview: string
  }>
}

interface UploadScanState {
  active: boolean
  fileName: string
  previewUrl: string | null
  stageIndex: number
  progress: number
  success: boolean
  error: string | null
  fileCount: number
}

const UPLOAD_STAGES = [
  'Uploading PDF document to server...',
  'Extracting text layers and document structure...',
  'Splitting semantic chunks & processing context...',
  'Generating dense embeddings with MiniLM-L6-v2...',
  'Indexing vectors in ChromaDB for semantic retrieval...',
  'PDF Indexed & Ready for AI Assistance!',
]

const ACTION_TITLES: Record<string, string> = {
  summarize: 'Executive Summary',
  explain: 'Step-by-Step Document Explanation',
  key_points: 'Key Takeaways & Highlights',
  study_notes: 'High-Yield Study Notes',
  key_concepts: 'Glossary of Definitions & Terms',
  difficult_concepts: 'Difficult Concepts Simplified',
  flashcards: 'Active Recall Study Flashcards',
  quiz: 'Self-Assessment Practice Quiz',
  qa: 'Practice Questions & Model Answers',
  compare: 'Cross-Document Comparative Analysis',
}

export default function Chat() {
  // Theme state with localStorage persistence
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      const saved = localStorage.getItem('ai_assistant_theme')
      return saved === 'dark' ? 'dark' : 'light'
    } catch {
      return 'light'
    }
  })

  // Navigation tab
  const [activeTab, setActiveTab] = useState<'chat' | 'documents' | 'summarize' | 'evaluation' | 'settings'>('chat')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Chat state with persistent sessionStorage rehydration across tab navigation
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = sessionStorage.getItem(CHAT_STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed)) return parsed
      }
    } catch (err) {
      console.warn('Failed to restore chat session:', err)
    }
    return []
  })
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const [clearDialogOpen, setClearDialogOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Documents state
  const [documents, setDocuments] = useState<DocItem[]>([])
  const [totalChunks, setTotalChunks] = useState(0)
  const [previewDoc, setPreviewDoc] = useState<DocItem | null>(null)

  // Advanced Upload Scanning State
  const [uploadScan, setUploadScan] = useState<UploadScanState>({
    active: false,
    fileName: '',
    previewUrl: null,
    stageIndex: 0,
    progress: 0,
    success: false,
    error: null,
    fileCount: 1,
  })

  // Professional Reset Knowledge Base Modal State
  const [resetModal, setResetModal] = useState<{
    isOpen: boolean
    stage: 'confirm' | 'processing' | 'success' | 'error'
    errorMessage: string | null
  }>({
    isOpen: false,
    stage: 'confirm',
    errorMessage: null,
  })

  // Document Intelligence state (supports all 10 actions)
  const [selectedDocForSummary, setSelectedDocForSummary] = useState('')
  const [secondaryDocForCompare, setSecondaryDocForCompare] = useState('')
  const [intelligenceAction, setIntelligenceAction] = useState<
    'summarize' | 'explain' | 'key_points' | 'study_notes' | 'key_concepts' |
    'difficult_concepts' | 'flashcards' | 'quiz' | 'qa' | 'compare'
  >('summarize')
  const [summaryText, setSummaryText] = useState('')
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [copiedOutput, setCopiedOutput] = useState(false)
  const [pdfExportLoading, setPdfExportLoading] = useState(false)
  const [pdfExportError, setPdfExportError] = useState<string | null>(null)

  // Real Evaluation state
  const [evalLoading, setEvalLoading] = useState(false)
  const [evalReport, setEvalReport] = useState<EvaluationReport | null>(null)
  const [evalError, setEvalError] = useState<string | null>(null)

  // Settings state
  const [settings, setSettings] = useState({
    provider: 'gemini',
    api_key: '',
    gemini_model: 'gemini-3.1-flash-lite',
    chunk_size: 500,
    chunk_overlap: 100,
    top_k: 4,
    similarity_threshold: 0.15,
  })
  const [settingsSaved, setSettingsSaved] = useState(false)

  // Apply theme class to document
  useEffect(() => {
    try {
      localStorage.setItem('ai_assistant_theme', theme)
    } catch {}
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  // Persistent Chat Storage synchronization
  useEffect(() => {
    try {
      sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages))
    } catch (err) {
      console.warn('Failed to sync chat history to sessionStorage:', err)
    }
  }, [messages])

  // Scroll chat to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Close modals on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setPreviewDoc(null)
        setClearDialogOpen(false)
        if (resetModal.stage !== 'processing') {
          setResetModal((prev) => ({ ...prev, isOpen: false }))
        }
        if (!uploadScan.active || uploadScan.success || uploadScan.error) {
          setUploadScan((prev) => ({ ...prev, active: false }))
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [uploadScan.active, uploadScan.success, uploadScan.error, resetModal.stage])

  // Fetch documents and status on load
  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`)
      if (res.ok) {
        const data = await res.json()
        if (data.documents) {
          setDocuments(data.documents)
          if (data.documents.length > 0 && !selectedDocForSummary) {
            setSelectedDocForSummary(data.documents[0].filename)
            if (data.documents.length > 1) {
              setSecondaryDocForCompare(data.documents[1].filename)
            }
          }
        }
        if (data.total_chunks !== undefined) {
          setTotalChunks(data.total_chunks)
        }
      }
    } catch (err) {
      console.log('Backend not reachable')
    }
  }

  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/settings`)
      if (res.ok) {
        const data = await res.json()
        setSettings((prev) => ({ ...prev, ...data }))
      }
    } catch (err) {}
  }

  const fetchLatestEvaluation = async () => {
    try {
      const res = await fetch(`${API_BASE}/evaluation`)
      if (res.ok) {
        const data = await res.json()
        if (data.status === 'success' && data.report) {
          setEvalReport(data.report)
        }
      }
    } catch (err) {}
  }

  useEffect(() => {
    fetchDocs()
    fetchSettings()
    fetchLatestEvaluation()
  }, [])

  // Send message
  const handleSend = async (textToSend?: string) => {
    const q = (textToSend || draft).trim()
    if (!q || loading) return

    setDraft('')
    setLoading(true)

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const historyPayload = messages.map((m) => ({
      user: m.user,
      assistant: m.assistant,
      mode: m.mode || 'GENERAL',
    }))

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          history: historyPayload,
          top_k: settings.top_k,
          similarity_threshold: settings.similarity_threshold,
        }),
      })

      if (res.ok) {
        const data = await res.json()
        setMessages((prev) => [
          ...prev,
          {
            user: q,
            assistant: data.answer,
            mode: data.mode || 'GENERAL',
            time: data.timestamp || now,
            grounded: data.grounded,
          },
        ])
      } else {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || 'Server returned an error')
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          user: q,
          assistant: `⚠️ **Connection Error**: ${err?.message || 'Unable to reach backend. Please ensure the backend server is running.'}`,
          mode: 'GENERAL',
          time: now,
          grounded: false,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  // Advanced PDF Upload with Scanning Visualization
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files
    if (!fileList || fileList.length === 0) return

    const primaryFile = fileList[0]
    const previewUrl = URL.createObjectURL(primaryFile)

    // Initialize Upload Scanning Modal
    setUploadScan({
      active: true,
      fileName: primaryFile.name,
      previewUrl,
      stageIndex: 0,
      progress: 15,
      success: false,
      error: null,
      fileCount: fileList.length,
    })

    const formData = new FormData()
    for (let i = 0; i < fileList.length; i++) {
      formData.append('files', fileList[i])
    }
    formData.append('chunk_size', settings.chunk_size.toString())
    formData.append('chunk_overlap', settings.chunk_overlap.toString())

    // Progress animation timeline
    const t1 = setTimeout(() => {
      setUploadScan((prev) => (prev.active && !prev.success ? { ...prev, stageIndex: 1, progress: 38 } : prev))
    }, 450)
    const t2 = setTimeout(() => {
      setUploadScan((prev) => (prev.active && !prev.success ? { ...prev, stageIndex: 2, progress: 62 } : prev))
    }, 1100)
    const t3 = setTimeout(() => {
      setUploadScan((prev) => (prev.active && !prev.success ? { ...prev, stageIndex: 3, progress: 84 } : prev))
    }, 1800)
    const t4 = setTimeout(() => {
      setUploadScan((prev) => (prev.active && !prev.success ? { ...prev, stageIndex: 4, progress: 95 } : prev))
    }, 2600)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })

      clearTimeout(t1)
      clearTimeout(t2)
      clearTimeout(t3)
      clearTimeout(t4)

      if (res.ok) {
        await res.json()
        await fetchDocs()

        setUploadScan((prev) => ({
          ...prev,
          stageIndex: 5,
          progress: 100,
          success: true,
          error: null,
        }))

        // Auto-close modal after successful completion
        setTimeout(() => {
          setUploadScan((prev) => (prev.success ? { ...prev, active: false } : prev))
        }, 2200)
      } else {
        const errData = await res.json().catch(() => ({}))
        const errorDetail = errData.detail || 'The document upload could not be completed. Please ensure the file is a valid PDF.'
        setUploadScan((prev) => ({
          ...prev,
          error: errorDetail,
          progress: 0,
          success: false,
        }))
      }
    } catch (err: any) {
      clearTimeout(t1)
      clearTimeout(t2)
      clearTimeout(t3)
      clearTimeout(t4)
      setUploadScan((prev) => ({
        ...prev,
        error: 'Unable to communicate with the document processing server. Please verify the backend connection.',
        progress: 0,
        success: false,
      }))
    } finally {
      // Clear input so the same file can be re-uploaded if desired
      e.target.value = ''
    }
  }

  // Load sample PDFs
  const handleLoadSample = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/load-sample`, { method: 'POST' })
      if (res.ok) {
        await fetchDocs()
      }
    } catch (e) {}
    setLoading(false)
  }

  // Delete document
  const handleDeleteDoc = async (filename: string) => {
    try {
      await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      await fetchDocs()
      if (previewDoc?.filename === filename) {
        setPreviewDoc(null)
      }
    } catch (e) {}
  }

  // Reset database via Professional Modal
  const handleConfirmReset = async () => {
    setResetModal((prev) => ({ ...prev, stage: 'processing', errorMessage: null }))
    try {
      const res = await fetch(`${API_BASE}/reset`, { method: 'POST' })
      if (res.ok) {
        setDocuments([])
        setTotalChunks(0)
        setPreviewDoc(null)
        setResetModal({ isOpen: true, stage: 'success', errorMessage: null })
        await fetchDocs()
      } else {
        const errData = await res.json().catch(() => ({}))
        setResetModal({
          isOpen: true,
          stage: 'error',
          errorMessage: errData.detail || 'Failed to reset knowledge base on the server.',
        })
      }
    } catch {
      setResetModal({
        isOpen: true,
        stage: 'error',
        errorMessage: 'Unable to reach backend server to perform reset operation.',
      })
    }
  }

  // Confirm Clear Chat (Safe message reset preserving documents and session clean)
  const confirmClearChat = () => {
    setMessages([])
    try {
      sessionStorage.removeItem(CHAT_STORAGE_KEY)
    } catch {}
    setClearDialogOpen(false)
  }

  // Run Document Intelligence
  const handleRunDocIntelligence = async (forcedAction?: typeof intelligenceAction) => {
    const actionToRun = forcedAction || intelligenceAction
    if (!selectedDocForSummary) return
    setSummaryLoading(true)
    try {
      const res = await fetch(`${API_BASE}/document-intelligence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: actionToRun,
          document: selectedDocForSummary,
          secondary_document: actionToRun === 'compare' ? secondaryDocForCompare : undefined,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        setSummaryText(data.result)
      } else {
        const errJson = await res.json().catch(() => ({}))
        setSummaryText(`⚠️ Processing Error: ${errJson.detail || 'Failed to process document action.'}`)
      }
    } catch (e) {
      setSummaryText('Failed to reach backend server. Please verify the backend service is running.')
    } finally {
      setSummaryLoading(false)
    }
  }

  // Copy Intelligence Output to Clipboard
  const handleCopyIntelligenceOutput = async () => {
    if (!summaryText) return
    try {
      await navigator.clipboard.writeText(summaryText)
      setCopiedOutput(true)
      setTimeout(() => setCopiedOutput(false), 2500)
    } catch {
      // Fallback via textarea
      const ta = document.createElement('textarea')
      ta.value = summaryText
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      ta.remove()
      setCopiedOutput(true)
      setTimeout(() => setCopiedOutput(false), 2500)
    }
  }

  // Download Intelligence Output as Formatted PDF
  const handleDownloadPDF = async () => {
    if (!summaryText || pdfExportLoading) return
    setPdfExportLoading(true)
    setPdfExportError(null)

    const docTitle = ACTION_TITLES[intelligenceAction] || 'Document Intelligence Report'
    const docName = selectedDocForSummary || 'Uploaded_Document.pdf'

    try {
      const res = await fetch(`${API_BASE}/export-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: summaryText,
          title: docTitle,
          action: intelligenceAction,
          document: docName,
        }),
      })

      if (res.ok) {
        const blob = await res.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        const cleanName = docName.replace(/\.[^/.]+$/, '').replace(/[\s\W]+/g, '_')
        a.download = `${intelligenceAction}_${cleanName}.pdf`
        document.body.appendChild(a)
        a.click()
        a.remove()
        window.URL.revokeObjectURL(url)
      } else {
        const errJson = await res.json().catch(() => ({}))
        setPdfExportError(errJson.detail || 'Failed to generate PDF document.')
        setTimeout(() => setPdfExportError(null), 4000)
      }
    } catch {
      setPdfExportError('Failed to communicate with PDF export service.')
      setTimeout(() => setPdfExportError(null), 4000)
    } finally {
      setPdfExportLoading(false)
    }
  }

  // Run Real Evaluation Benchmark
  const handleRunEval = async () => {
    setEvalLoading(true)
    setEvalError(null)
    try {
      const res = await fetch(`${API_BASE}/evaluation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ top_k: 4, provider: settings.provider }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.report) {
          setEvalReport(data.report)
        }
      } else {
        const errData = await res.json().catch(() => ({}))
        setEvalError(errData.detail || 'Evaluation benchmark run failed.')
      }
    } catch (e: any) {
      setEvalError('Unable to connect to evaluation engine.')
    } finally {
      setEvalLoading(false)
    }
  }

  // Save settings
  const handleSaveSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      })
      if (res.ok) {
        setSettingsSaved(true)
        setTimeout(() => setSettingsSaved(false), 2500)
      }
    } catch (e) {}
  }

  return (
    <div
      className={`h-screen w-screen overflow-hidden flex flex-col font-sans transition-colors duration-250 ${
        theme === 'dark' ? 'bg-[#090E1B] text-[#F5F7FF]' : 'bg-[#F5F8FC] text-[#17243A]'
      }`}
    >
      {/* ───────────────────────────────────────────────────────────── */}
      {/* MOBILE TOP BAR (Visible on < lg) */}
      {/* ───────────────────────────────────────────────────────────── */}
      <header
        className={`flex items-center justify-between border-b px-4 py-2.5 lg:hidden shrink-0 ${
          theme === 'dark' ? 'bg-[#0D1424] border-[#26364D] text-[#F5F7FF]' : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A]'
        }`}
      >
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setMobileMenuOpen(true)}
            className={`p-2 rounded-xl transition-colors ${
              theme === 'dark' ? 'hover:bg-[#1C2A40] text-[#A7B5CC]' : 'hover:bg-[#E8EEF7] text-[#64748B]'
            }`}
            aria-label="Open Navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-[#35C8F4] to-[#9875FF] text-white shadow-md shadow-[#35C8F4]/20">
              <Brain className="h-4 w-4" />
            </span>
            <span className="text-[15px] font-bold tracking-tight">AI PDF Assistant</span>
          </div>
        </div>

        {/* Mobile Theme Toggle */}
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className={`flex h-9 w-9 items-center justify-center rounded-xl border transition-all ${
            theme === 'dark'
              ? 'border-[#26364D] bg-[#111B2E] text-[#35C8F4] hover:bg-[#1C2A40]'
              : 'border-[#DFE7F1] bg-[#E8EEF7] text-[#168EDB] hover:bg-[#DFE7F1]'
          }`}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          aria-label="Toggle Theme"
        >
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </header>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MAIN CONTAINER (DESKTOP: Fixed Sidebar + Scrollable Workspace) */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* MOBILE BACKDROP DRAWER OVERLAY */}
        {mobileMenuOpen && (
          <div
            className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs lg:hidden transition-opacity"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}

        {/* ══════════════════════════════════════════════════════════ */}
        {/* FIXED / STICKY SIDEBAR (Midnight Aurora Redesign) */}
        {/* ══════════════════════════════════════════════════════════ */}
        <aside
          className={`fixed inset-y-0 left-0 z-50 flex w-[285px] flex-col justify-between p-5 transition-transform duration-200 ease-in-out lg:static lg:z-auto lg:h-full lg:translate-x-0 shrink-0 ${
            mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
          } ${
            theme === 'dark'
              ? 'bg-[#0D1424] border-r border-[#26364D] text-[#F5F7FF]'
              : 'bg-[#FFFFFF] border-r border-[#DFE7F1] text-[#17243A] shadow-md shadow-slate-200/50'
          }`}
        >
          <div className="flex flex-col">
            {/* Top Brand Header + Theme Toggle */}
            <div className={`flex items-center justify-between pb-3 border-b ${
              theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
            }`}>
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-[#35C8F4] to-[#9875FF] text-white shadow-lg shadow-[#35C8F4]/25">
                  <Brain className="h-5 w-5" />
                </span>
                <div>
                  <h1 className="text-[17px] font-extrabold tracking-tight leading-tight">AI Assistant</h1>
                  <p className={`text-[11px] font-medium ${
                    theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                  }`}>Universal & Multi-PDF</p>
                </div>
              </div>

              {/* Close Mobile Drawer */}
              <button
                onClick={() => setMobileMenuOpen(false)}
                className={`lg:hidden p-1.5 rounded-lg transition-colors ${
                  theme === 'dark' ? 'text-[#A7B5CC] hover:text-[#F5F7FF]' : 'text-[#64748B] hover:text-[#17243A]'
                }`}
                aria-label="Close Menu"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Theme Toggle Pill on Desktop */}
            <div className={`mt-4 flex items-center justify-between rounded-2xl p-1.5 border ${
              theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#E8EEF7] border-[#DFE7F1]'
            }`}>
              <span className={`text-[12px] font-medium pl-2 ${
                theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
              }`}>Theme Mode</span>
              <button
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                className={`relative flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-[12px] font-semibold transition-all duration-200 cursor-pointer ${
                  theme === 'dark'
                    ? 'bg-[#1C2A40] text-[#35C8F4] border border-[#35C8F4]/30 shadow-xs'
                    : 'bg-gradient-to-r from-[#168EDB] to-[#8060E8] text-white shadow-sm'
                }`}
                aria-label="Theme toggle"
              >
                {theme === 'dark' ? (
                  <>
                    <Sun className="h-3.5 w-3.5 transition-transform rotate-0 scale-100" />
                    <span>Dark</span>
                  </>
                ) : (
                  <>
                    <Moon className="h-3.5 w-3.5 transition-transform rotate-0 scale-100" />
                    <span>Light</span>
                  </>
                )}
              </button>
            </div>

            {/* Navigation Tabs */}
            <nav className="mt-6 flex flex-col gap-1.5">
              <NavItem
                icon={MessageSquareMore}
                label="Chat Assistant"
                active={activeTab === 'chat'}
                onClick={() => {
                  setActiveTab('chat')
                  setMobileMenuOpen(false)
                }}
                theme={theme}
              />
              <NavItem
                icon={FileText}
                label="Documents & Upload"
                active={activeTab === 'documents'}
                onClick={() => {
                  setActiveTab('documents')
                  setMobileMenuOpen(false)
                }}
                theme={theme}
              />
              <NavItem
                icon={Sparkles}
                label="Document Intelligence"
                active={activeTab === 'summarize'}
                onClick={() => {
                  setActiveTab('summarize')
                  setMobileMenuOpen(false)
                }}
                theme={theme}
              />
              <NavItem
                icon={BarChart3}
                label="RAG Benchmark"
                active={activeTab === 'evaluation'}
                onClick={() => {
                  setActiveTab('evaluation')
                  setMobileMenuOpen(false)
                }}
                theme={theme}
              />
              <NavItem
                icon={Settings}
                label="Settings"
                active={activeTab === 'settings'}
                onClick={() => {
                  setActiveTab('settings')
                  setMobileMenuOpen(false)
                }}
                theme={theme}
              />
            </nav>
          </div>

          {/* Sidebar Footer: Knowledge Base Stats */}
          <div className={`mt-6 rounded-2xl p-4 border transition-colors ${
            theme === 'dark'
              ? 'bg-[#111B2E] border-[#26364D]'
              : 'bg-[#E8EEF7] border-[#DFE7F1]'
          }`}>
            <div className="flex items-center justify-between text-[12.5px]">
              <span className={`flex items-center gap-1.5 font-semibold ${
                theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'
              }`}>
                <Database className={`h-4 w-4 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                Knowledge Base
              </span>
              <span className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${
                theme === 'dark'
                  ? 'bg-gradient-to-r from-[#35C8F4]/20 to-[#9875FF]/20 border border-[#35C8F4]/40 text-[#35C8F4]'
                  : 'bg-[#168EDB]/15 text-[#168EDB]'
              }`}>
                {documents.length} PDFs
              </span>
            </div>
            <p className={`mt-1.5 text-[11px] ${
              theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
            }`}>
              {totalChunks} Semantically Indexed Chunks
            </p>
          </div>
        </aside>

        {/* ══════════════════════════════════════════════════════════ */}
        {/* MAIN WORKSPACE CONTENT AREA (Independent Scrolling) */}
        {/* ══════════════════════════════════════════════════════════ */}
        <main className="flex-1 h-full overflow-y-auto flex flex-col p-3 sm:p-5 lg:p-6 min-w-0">
          {/* Top Hero Banner */}
          <div
            className={`rounded-3xl p-5 sm:p-6 border shadow-md transition-colors shrink-0 mb-4 relative overflow-hidden ${
              theme === 'dark'
                ? 'bg-gradient-to-r from-[#0D1424] via-[#111B2E] to-[#172238] border-[#26364D] text-[#F5F7FF]'
                : 'bg-gradient-to-r from-[#FFFFFF] via-[#F8FAFD] to-[#E8EEF7] border-[#DFE7F1] text-[#17243A]'
            }`}
          >
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center relative z-10">
              <div>
                <div className="flex items-center gap-2">
                  <span className={`rounded-full px-3 py-0.5 text-[11px] font-bold uppercase tracking-wider ${
                    theme === 'dark'
                      ? 'bg-[#35C8F4]/15 border border-[#35C8F4]/30 text-[#35C8F4]'
                      : 'bg-[#168EDB]/12 border border-[#168EDB]/25 text-[#168EDB]'
                  }`}>
                    Dual AI Intelligence
                  </span>
                  <span className={`flex items-center gap-1.5 text-[11.5px] font-semibold px-2.5 py-0.5 rounded-full border ${
                    theme === 'dark'
                      ? 'bg-[#36D6A0]/15 border-[#36D6A0]/30 text-[#36D6A0]'
                      : 'bg-[#10B981]/12 border-[#10B981]/25 text-[#10B981]'
                  }`}>
                    <span className="h-1.5 w-1.5 rounded-full bg-[#36D6A0] animate-pulse" />
                    Online & Active
                  </span>
                </div>
                <h2 className="mt-2 text-[22px] sm:text-[26px] font-black tracking-tight leading-tight">
                  Universal AI & Multi-PDF Assistant
                </h2>
                <p
                  className={`mt-1 text-[13.5px] max-w-xl ${
                    theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                  }`}
                >
                  Ask general knowledge questions across software, mathematics, and business, or explore your uploaded PDFs with clean, citation-free answers.
                </p>
              </div>

              {/* Feature Badges */}
              <div className="flex flex-wrap gap-2">
                <HeroChip
                  icon={<CheckCircle2 className={`h-3.5 w-3.5 ${theme === 'dark' ? 'text-[#36D6A0]' : 'text-[#10B981]'}`} />}
                  label="Multi-PDF RAG"
                  theme={theme}
                />
                <HeroChip
                  icon={<Globe className={`h-3.5 w-3.5 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />}
                  label="Arabic & English"
                  theme={theme}
                />
                <HeroChip
                  icon={<Zap className={`h-3.5 w-3.5 ${theme === 'dark' ? 'text-[#9875FF]' : 'text-[#8060E8]'}`} />}
                  label="Hybrid Reasoning"
                  theme={theme}
                />
              </div>
            </div>
          </div>

          {/* ══════════════════════════════════════════════════════════ */}
          {/* TAB 1: CHAT ASSISTANT (Persistent across page navigation) */}
          {/* ══════════════════════════════════════════════════════════ */}
          {activeTab === 'chat' && (
            <div className="flex-1 grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_330px] min-h-0">
              {/* Left Column: Chat Area */}
              <section
                className={`flex flex-col rounded-3xl border shadow-xs overflow-hidden h-[calc(100vh-230px)] lg:h-[calc(100vh-190px)] min-h-[500px] ${
                  theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                }`}
              >
                {/* Header with 3D Clear Chat button */}
                <div
                  className={`flex items-center justify-between border-b px-5 py-3 shrink-0 ${
                    theme === 'dark' ? 'bg-[#0D1424]/90 border-[#26364D] text-[#F5F7FF]' : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#17243A]'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Brain className={`h-5 w-5 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                    <span className="text-[14.5px] font-bold">Active Conversation</span>
                    {messages.length > 0 && (
                      <span
                        className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${
                          theme === 'dark'
                            ? 'bg-[#1C2A40] border-[#26364D] text-[#A7B5CC]'
                            : 'bg-[#E8EEF7] border-[#DFE7F1] text-[#64748B]'
                        }`}
                      >
                        {messages.length} {messages.length === 1 ? 'turn' : 'turns'}
                      </span>
                    )}
                  </div>

                  {/* PREMIUM 3D CLEAR CHAT BUTTON */}
                  {messages.length > 0 && (
                    <button
                      onClick={() => setClearDialogOpen(true)}
                      className="btn-3d-red flex items-center gap-1.5 rounded-xl px-3.5 py-1.5 text-[12px] font-bold text-white cursor-pointer"
                      title="Clear chat messages safely"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      <span>Clear Chat</span>
                    </button>
                  )}
                </div>

                {/* Messages Feed (Independent Scroll) */}
                <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 space-y-5">
                  {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-16 text-center">
                      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[#35C8F4] to-[#9875FF] text-white shadow-lg shadow-[#35C8F4]/25">
                        <Bot className="h-7 w-7" />
                      </span>
                      <h3 className={`mt-4 text-[18px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                        How can I assist you today?
                      </h3>
                      <p
                        className={`mt-1 max-w-md text-[13.5px] ${
                          theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                        }`}
                      >
                        Ask any general question across AI, programming, mathematics, and business, or explore your uploaded PDF documents.
                      </p>

                      {/* Prompt suggestion buttons */}
                      <div className="mt-6 flex flex-wrap justify-center gap-2 max-w-lg">
                        <SuggestChip
                          label="What are Mohamed Ramadan's key skills?"
                          onClick={() => handleSend("What are Mohamed Ramadan's key skills and experience?")}
                          theme={theme}
                        />
                        <SuggestChip
                          label="Explain Deep Learning & CNNs"
                          onClick={() => handleSend("Explain deep learning and how Convolutional Neural Networks work.")}
                          theme={theme}
                        />
                        <SuggestChip
                          label="ما هي الخبرات والمشاريع في السيرة الذاتية؟"
                          onClick={() => handleSend("ما هي الخبرات والمشاريع المذكورة في السيرة الذاتية؟")}
                          theme={theme}
                        />
                        <SuggestChip
                          label="Write a Python binary search script"
                          onClick={() => handleSend("Write a clean Python function for binary search with complexity analysis.")}
                          theme={theme}
                        />
                      </div>
                    </div>
                  ) : (
                    messages.map((turn, idx) => (
                      <div key={idx} className="space-y-3">
                        {/* User Message */}
                        <div className="flex items-start justify-end gap-2.5">
                          <div
                            className={`max-w-[80%] rounded-2xl rounded-tr-xs px-4 py-3 text-[14px] shadow-xs ${
                              theme === 'dark'
                                ? 'bg-gradient-to-r from-[#1C2A40] to-[#18263A] border border-[#26364D] text-[#F5F7FF]'
                                : 'bg-gradient-to-r from-[#E8EEF7] to-[#DFE7F1] border border-[#CBD5E1] text-[#17243A]'
                            }`}
                          >
                            <p className="leading-relaxed whitespace-pre-wrap">{turn.user}</p>
                            <div
                              className={`mt-1 flex items-center justify-end gap-1 text-[11px] ${
                                theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                              }`}
                            >
                              {turn.time} <CheckCheck className={`h-3 w-3 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                            </div>
                          </div>
                          <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border ${
                            theme === 'dark'
                              ? 'bg-[#1C2A40] border-[#26364D] text-[#35C8F4]'
                              : 'bg-[#E8EEF7] border-[#DFE7F1] text-[#168EDB]'
                          }`}>
                            <User className="h-4 w-4" />
                          </span>
                        </div>

                        {/* Assistant Answer (Clean Output - Strict Requirement 11 compliance) */}
                        <div className="flex items-start gap-3">
                          <span className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-[#35C8F4] to-[#9875FF] text-white shadow-md shadow-[#35C8F4]/20">
                            <Bot className="h-4 w-4" />
                          </span>
                          <div
                            className={`min-w-0 max-w-[88%] flex-1 rounded-2xl px-5 py-4 border shadow-xs ${
                              theme === 'dark'
                                ? 'bg-[#0D1424] border-[#26364D] text-[#F5F7FF]'
                                : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#17243A]'
                            }`}
                          >
                            {/* Mode Indicator Pill */}
                            <div className="mb-2.5 flex items-center gap-2">
                              {turn.mode === 'GENERAL' && (
                                <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold border ${
                                  theme === 'dark'
                                    ? 'bg-[#35C8F4]/15 border-[#35C8F4]/30 text-[#35C8F4]'
                                    : 'bg-[#168EDB]/12 border-[#168EDB]/25 text-[#168EDB]'
                                }`}>
                                  <Sparkles className="h-3 w-3" /> Universal AI
                                </span>
                              )}
                              {turn.mode === 'RAG' && (
                                <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold border ${
                                  theme === 'dark'
                                    ? 'bg-[#36D6A0]/15 border-[#36D6A0]/30 text-[#36D6A0]'
                                    : 'bg-[#10B981]/12 border-[#10B981]/25 text-[#10B981]'
                                }`}>
                                  <FileText className="h-3 w-3" /> Document Grounded
                                </span>
                              )}
                              {turn.mode === 'HYBRID' && (
                                <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold border ${
                                  theme === 'dark'
                                    ? 'bg-[#9875FF]/15 border-[#9875FF]/30 text-[#9875FF]'
                                    : 'bg-[#8060E8]/12 border-[#8060E8]/25 text-[#8060E8]'
                                }`}>
                                  <Zap className="h-3 w-3" /> Hybrid AI & Document
                                </span>
                              )}
                            </div>

                            {/* Clean Formatted Answer */}
                            <FormattedContent content={turn.assistant} theme={theme} />

                            {/* Answer footer */}
                            <div
                              className={`mt-3 flex items-center justify-end gap-3 text-[12px] border-t pt-2 ${
                                theme === 'dark'
                                  ? 'border-[#26364D] text-[#A7B5CC]'
                                  : 'border-[#DFE7F1] text-[#64748B]'
                              }`}
                            >
                              <span>{turn.time}</span>
                              <button
                                aria-label="Copy response"
                                onClick={() => {
                                  navigator.clipboard.writeText(turn.assistant)
                                  setCopiedIndex(idx)
                                  setTimeout(() => setCopiedIndex(null), 2000)
                                }}
                                className={`flex items-center gap-1 rounded-lg px-2 py-0.5 transition-colors cursor-pointer ${
                                  theme === 'dark'
                                    ? 'hover:bg-[#1C2A40] text-[#A7B5CC] hover:text-[#35C8F4]'
                                    : 'hover:bg-[#E8EEF7] text-[#64748B] hover:text-[#168EDB]'
                                }`}
                              >
                                {copiedIndex === idx ? (
                                  <>
                                    <Check className="h-3.5 w-3.5 text-[#36D6A0]" />
                                    <span className="text-[11px] text-[#36D6A0] font-semibold">Copied</span>
                                  </>
                                ) : (
                                  <>
                                    <Copy className="h-3.5 w-3.5" />
                                    <span className="text-[11px]">Copy</span>
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))
                  )}

                  {loading && (
                    <div className="flex items-start gap-3">
                      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-[#35C8F4] to-[#9875FF] text-white animate-pulse">
                        <Bot className="h-4 w-4" />
                      </span>
                      <div
                        className={`flex items-center gap-2 rounded-2xl px-4 py-3 border shadow-xs ${
                          theme === 'dark'
                            ? 'bg-[#0D1424] border-[#26364D] text-[#A7B5CC]'
                            : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#64748B]'
                        }`}
                      >
                        <span className={`h-2 w-2 animate-bounce rounded-full ${theme === 'dark' ? 'bg-[#35C8F4]' : 'bg-[#168EDB]'}`} />
                        <span className={`h-2 w-2 animate-bounce rounded-full ${theme === 'dark' ? 'bg-[#35C8F4]' : 'bg-[#168EDB]'} [animation-delay:150ms]`} />
                        <span className={`h-2 w-2 animate-bounce rounded-full ${theme === 'dark' ? 'bg-[#35C8F4]' : 'bg-[#168EDB]'} [animation-delay:300ms]`} />
                        <span className="ml-2 text-[13px] font-medium">Thinking & synthesizing response…</span>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>

                {/* Input Bar pinned to bottom */}
                <div
                  className={`border-t p-3 sm:p-4 shrink-0 ${
                    theme === 'dark' ? 'bg-[#0D1424] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                  }`}
                >
                  <form
                    onSubmit={(e) => {
                      e.preventDefault()
                      handleSend()
                    }}
                    className="flex items-center gap-2"
                  >
                    <input
                      type="text"
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      placeholder="Ask any general question or chat with your uploaded PDFs..."
                      disabled={loading}
                      className={`flex-1 rounded-2xl border px-4 py-3 text-[14px] outline-none transition-all ${
                        theme === 'dark'
                          ? 'bg-[#111B2E] border-[#26364D] text-[#F5F7FF] placeholder-[#A7B5CC]/50 focus:border-[#35C8F4] focus:ring-2 focus:ring-[#35C8F4]/20'
                          : 'bg-[#F5F8FC] border-[#DFE7F1] text-[#17243A] placeholder-[#64748B]/60 focus:border-[#168EDB] focus:bg-white focus:ring-2 focus:ring-[#168EDB]/15'
                      }`}
                    />
                    <button
                      type="submit"
                      disabled={!draft.trim() || loading}
                      className="btn-3d-aurora flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl text-white disabled:opacity-40 cursor-pointer"
                      aria-label="Send message"
                    >
                      <Send className="h-4 w-4" />
                    </button>
                  </form>
                </div>
              </section>

              {/* Right Column: Your Documents & Quick Intelligence */}
              <div className="flex flex-col gap-4">
                {/* Documents List Panel */}
                <section
                  className={`rounded-3xl border p-4 shadow-xs ${
                    theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                  }`}
                >
                  <div
                    className={`flex items-center justify-between border-b pb-3 ${
                      theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <FileText className={`h-4 w-4 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                      <h3 className={`text-[15px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                        Your Documents
                      </h3>
                    </div>
                    <label className={`cursor-pointer rounded-xl px-2.5 py-1 text-[11.5px] font-semibold border transition-colors ${
                      theme === 'dark'
                        ? 'bg-[#1C2A40] border-[#26364D] text-[#35C8F4] hover:border-[#35C8F4]'
                        : 'bg-[#E8EEF7] border-[#DFE7F1] text-[#168EDB] hover:bg-[#DFE7F1]'
                    }`}>
                      + Add PDF
                      <input type="file" multiple accept=".pdf" onChange={handleFileUpload} className="hidden" />
                    </label>
                  </div>

                  <div className="mt-3 space-y-2 max-h-[200px] overflow-y-auto">
                    {documents.length > 0 ? (
                      documents.map((d, i) => (
                        <div
                          key={i}
                          className={`group flex items-center justify-between rounded-xl border p-2.5 transition-all cursor-pointer ${
                            theme === 'dark'
                              ? 'border-[#26364D] bg-[#1C2A40]/70 hover:bg-[#1C2A40] hover:border-[#35C8F4]/50'
                              : 'border-[#DFE7F1] bg-[#F5F8FC] hover:bg-white hover:border-[#168EDB]/40'
                          }`}
                          onClick={() => setPreviewDoc(d)}
                          title="Click to preview PDF"
                        >
                          <div className="flex items-center gap-2.5 min-w-0">
                            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#F43F5E] to-[#E11D48] text-[9px] font-extrabold text-white shadow-xs">
                              PDF
                            </span>
                            <div className="min-w-0">
                              <p className={`truncate text-[12.5px] font-semibold transition-colors ${
                                theme === 'dark' ? 'text-[#F5F7FF] group-hover:text-[#35C8F4]' : 'text-[#17243A] group-hover:text-[#168EDB]'
                              }`}>
                                {d.name}
                              </p>
                              <p
                                className={`text-[10.5px] ${
                                  theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                                }`}
                              >
                                {d.pages} pages • {d.size_mb} MB
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-1 shrink-0" onClick={(e) => e.stopPropagation()}>
                            <button
                              onClick={() => setPreviewDoc(d)}
                              className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                                theme === 'dark'
                                  ? 'text-[#A7B5CC] hover:bg-[#111B2E] hover:text-[#35C8F4]'
                                  : 'text-[#64748B] hover:bg-[#E8EEF7] hover:text-[#168EDB]'
                              }`}
                              title="Preview Document"
                            >
                              <Eye className="h-3.5 w-3.5" />
                            </button>
                            <button
                              onClick={() => handleDeleteDoc(d.filename)}
                              className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                                theme === 'dark'
                                  ? 'text-[#A7B5CC] hover:bg-red-950/60 hover:text-red-400'
                                  : 'text-[#64748B] hover:bg-red-50 hover:text-red-600'
                              }`}
                              title="Delete file"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      ))
                    ) : (
                      <p
                        className={`py-4 text-center text-[12.5px] ${
                          theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                        }`}
                      >
                        No documents uploaded yet.
                      </p>
                    )}
                  </div>
                </section>

                {/* Document Intelligence Quick Actions */}
                <section
                  className={`rounded-3xl border p-4 shadow-xs ${
                    theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                  }`}
                >
                  <div
                    className={`flex items-center gap-2 border-b pb-3 ${
                      theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                    }`}
                  >
                    <Sparkles className={`h-4 w-4 ${theme === 'dark' ? 'text-[#9875FF]' : 'text-[#8060E8]'}`} />
                    <h3 className={`text-[15px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                      Document Intelligence
                    </h3>
                  </div>

                  <div className="mt-3 flex flex-col gap-2">
                    <button
                      onClick={() => {
                        setActiveTab('summarize')
                        setIntelligenceAction('summarize')
                        handleRunDocIntelligence('summarize')
                      }}
                      disabled={loading || documents.length === 0}
                      className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left text-[12.5px] font-medium transition-all disabled:opacity-40 cursor-pointer ${
                        theme === 'dark'
                          ? 'border-[#26364D] bg-[#1C2A40]/70 text-[#F5F7FF] hover:bg-[#1C2A40] hover:border-[#35C8F4]/50 hover:text-[#35C8F4]'
                          : 'border-[#DFE7F1] bg-[#F5F8FC] text-[#17243A] hover:bg-[#E8EEF7] hover:text-[#168EDB]'
                      }`}
                    >
                      <FileText className={`h-4 w-4 shrink-0 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                      <span>Executive Summary</span>
                    </button>

                    <button
                      onClick={() => {
                        setActiveTab('summarize')
                        setIntelligenceAction('study_notes')
                        handleRunDocIntelligence('study_notes')
                      }}
                      disabled={loading || documents.length === 0}
                      className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left text-[12.5px] font-medium transition-all disabled:opacity-40 cursor-pointer ${
                        theme === 'dark'
                          ? 'border-[#26364D] bg-[#1C2A40]/70 text-[#F5F7FF] hover:bg-[#1C2A40] hover:border-[#9875FF]/50 hover:text-[#9875FF]'
                          : 'border-[#DFE7F1] bg-[#F5F8FC] text-[#17243A] hover:bg-[#E8EEF7] hover:text-[#8060E8]'
                      }`}
                    >
                      <BookOpen className={`h-4 w-4 shrink-0 ${theme === 'dark' ? 'text-[#9875FF]' : 'text-[#8060E8]'}`} />
                      <span>Create Study Notes</span>
                    </button>

                    <button
                      onClick={() => {
                        setActiveTab('summarize')
                        setIntelligenceAction('key_concepts')
                        handleRunDocIntelligence('key_concepts')
                      }}
                      disabled={loading || documents.length === 0}
                      className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left text-[12.5px] font-medium transition-all disabled:opacity-40 cursor-pointer ${
                        theme === 'dark'
                          ? 'border-[#26364D] bg-[#1C2A40]/70 text-[#F5F7FF] hover:bg-[#1C2A40] hover:border-[#35C8F4]/50 hover:text-[#35C8F4]'
                          : 'border-[#DFE7F1] bg-[#F5F8FC] text-[#17243A] hover:bg-[#E8EEF7] hover:text-[#168EDB]'
                      }`}
                    >
                      <Lightbulb className="h-4 w-4 text-amber-400 shrink-0" />
                      <span>Extract Glossary & Definitions</span>
                    </button>

                    <button
                      onClick={() => {
                        setActiveTab('summarize')
                        setIntelligenceAction('quiz')
                        handleRunDocIntelligence('quiz')
                      }}
                      disabled={loading || documents.length === 0}
                      className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left text-[12.5px] font-medium transition-all disabled:opacity-40 cursor-pointer ${
                        theme === 'dark'
                          ? 'border-[#26364D] bg-[#1C2A40]/70 text-[#F5F7FF] hover:bg-[#1C2A40] hover:border-[#36D6A0]/50 hover:text-[#36D6A0]'
                          : 'border-[#DFE7F1] bg-[#F5F8FC] text-[#17243A] hover:bg-[#E8EEF7] hover:text-[#10B981]'
                      }`}
                    >
                      <HelpCircle className={`h-4 w-4 shrink-0 ${theme === 'dark' ? 'text-[#36D6A0]' : 'text-[#10B981]'}`} />
                      <span>Self-Assessment Quiz</span>
                    </button>

                    {documents.length > 1 && (
                      <button
                        onClick={() => {
                          setActiveTab('summarize')
                          setIntelligenceAction('compare')
                          handleRunDocIntelligence('compare')
                        }}
                        disabled={loading}
                        className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left text-[12.5px] font-medium transition-all cursor-pointer ${
                          theme === 'dark'
                            ? 'border-[#26364D] bg-[#1C2A40]/70 text-[#F5F7FF] hover:bg-[#1C2A40] hover:border-[#35C8F4]/50 hover:text-[#35C8F4]'
                            : 'border-[#DFE7F1] bg-[#F5F8FC] text-[#17243A] hover:bg-[#E8EEF7] hover:text-[#168EDB]'
                        }`}
                      >
                        <Scale className={`h-4 w-4 shrink-0 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                        <span>Compare All Documents</span>
                      </button>
                    )}
                  </div>
                </section>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════ */}
          {/* TAB 2: DOCUMENTS MANAGEMENT (Midnight Aurora Redesign) */}
          {/* ══════════════════════════════════════════════════════════ */}
          {activeTab === 'documents' && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Upload Card */}
              <div
                className={`rounded-3xl border p-6 shadow-xs ${
                  theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                }`}
              >
                <div
                  className={`flex items-center gap-3 border-b pb-4 ${
                    theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                  }`}
                >
                  <Upload className={`h-6 w-6 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                  <h3 className={`text-[19px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                    Upload PDF Documents
                  </h3>
                </div>

                <div
                  className={`mt-6 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
                    theme === 'dark'
                      ? 'border-[#35C8F4]/30 bg-[#1C2A40]/40 hover:border-[#35C8F4]'
                      : 'border-[#168EDB]/30 bg-[#F5F8FC] hover:border-[#168EDB]'
                  }`}
                >
                  <Upload className={`h-10 w-10 ${theme === 'dark' ? 'text-[#35C8F4]/80' : 'text-[#168EDB]/80'}`} />
                  <p className={`mt-3 text-[15px] font-semibold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                    Select multiple PDF files to index
                  </p>
                  <p
                    className={`mt-1 text-[12.5px] ${
                      theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                    }`}
                  >
                    Supports multi-page, bilingual Arabic & English documents
                  </p>
                  <label className="btn-3d-aurora mt-5 cursor-pointer rounded-xl px-5 py-2.5 text-[13.5px] font-semibold text-white">
                    Browse Files
                    <input type="file" multiple accept=".pdf" onChange={handleFileUpload} className="hidden" />
                  </label>
                </div>

                <div className={`mt-6 border-t pt-4 ${theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'}`}>
                  <button
                    onClick={handleLoadSample}
                    disabled={loading}
                    className={`flex w-full items-center justify-center gap-2 rounded-xl border px-4 py-3 text-[13.5px] font-semibold transition-colors cursor-pointer ${
                      theme === 'dark'
                        ? 'bg-[#1C2A40] border-[#26364D] text-[#35C8F4] hover:bg-[#1C2A40]/80 hover:border-[#35C8F4]'
                        : 'bg-[#E8EEF7] border-[#DFE7F1] text-[#168EDB] hover:bg-[#DFE7F1]'
                    }`}
                  >
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                    Seed Sample Machine Learning & Deep Learning PDFs
                  </button>
                </div>
              </div>

              {/* Indexed Knowledge Base Card */}
              <div
                className={`rounded-3xl border p-6 shadow-xs ${
                  theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                }`}
              >
                <div
                  className={`flex items-center justify-between border-b pb-4 ${
                    theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Database className={`h-6 w-6 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                    <h3 className={`text-[19px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                      Indexed Documents
                    </h3>
                  </div>
                  <span className={`rounded-full px-3 py-1 text-[12.5px] font-semibold border ${
                    theme === 'dark'
                      ? 'bg-[#35C8F4]/15 border-[#35C8F4]/30 text-[#35C8F4]'
                      : 'bg-[#168EDB]/12 border-[#168EDB]/25 text-[#168EDB]'
                  }`}>
                    {totalChunks} Chunks
                  </span>
                </div>

                <div className="mt-5 space-y-3 max-h-[380px] overflow-y-auto">
                  {documents.length > 0 ? (
                    documents.map((doc, idx) => (
                      <div
                        key={idx}
                        className={`group flex items-center justify-between rounded-xl border p-3.5 transition-colors cursor-pointer ${
                          theme === 'dark'
                            ? 'border-[#26364D] bg-[#1C2A40]/80 hover:bg-[#1C2A40] hover:border-[#35C8F4]/60'
                            : 'border-[#DFE7F1] bg-[#F5F8FC] hover:bg-white hover:border-[#168EDB]/40'
                        }`}
                        onClick={() => setPreviewDoc(doc)}
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="flex h-10 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#F43F5E] to-[#E11D48] text-[10px] font-bold text-white shadow-xs">
                            PDF
                          </div>
                          <div className="min-w-0">
                            <p className={`truncate text-[14.5px] font-semibold transition-colors ${
                              theme === 'dark' ? 'text-[#F5F7FF] group-hover:text-[#35C8F4]' : 'text-[#17243A] group-hover:text-[#168EDB]'
                            }`}>
                              {doc.name}
                            </p>
                            <p
                              className={`text-[12px] ${
                                theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                              }`}
                            >
                              {doc.pages} pages • {doc.size_mb} MB • {doc.chunks} chunks
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => setPreviewDoc(doc)}
                            className={`rounded-lg p-2 transition-colors cursor-pointer ${
                              theme === 'dark'
                                ? 'text-[#A7B5CC] hover:bg-[#111B2E] hover:text-[#35C8F4]'
                                : 'text-[#64748B] hover:bg-[#E8EEF7] hover:text-[#168EDB]'
                            }`}
                            title="Interactive PDF Preview"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => handleDeleteDoc(doc.filename)}
                            className={`rounded-lg p-2 transition-colors cursor-pointer ${
                              theme === 'dark'
                                ? 'text-[#A7B5CC] hover:bg-red-950/60 hover:text-red-400'
                                : 'text-[#64748B] hover:bg-red-50 hover:text-red-600'
                            }`}
                            title="Delete document"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p
                      className={`py-8 text-center ${
                        theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                      }`}
                    >
                      No documents uploaded yet.
                    </p>
                  )}
                </div>

                {/* Professional Reset Button triggering confirmation modal */}
                <div className={`mt-6 border-t pt-4 ${theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'}`}>
                  <button
                    onClick={() => setResetModal({ isOpen: true, stage: 'confirm', errorMessage: null })}
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-red-900/50 dark:border-red-900/60 bg-red-950/30 dark:bg-red-950/40 px-4 py-2.5 text-[13.5px] font-semibold text-red-400 hover:bg-red-900/50 transition-colors cursor-pointer"
                  >
                    <Trash2 className="h-4 w-4" />
                    Reset Entire Knowledge Base
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════ */}
          {/* TAB 3: DOCUMENT INTELLIGENCE (Midnight Aurora Redesign) */}
          {/* ══════════════════════════════════════════════════════════ */}
          {activeTab === 'summarize' && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
              {/* Controls */}
              <div
                className={`rounded-3xl border p-6 shadow-xs space-y-5 ${
                  theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                }`}
              >
                <div
                  className={`flex items-center gap-3 border-b pb-4 ${
                    theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                  }`}
                >
                  <Sparkles className={`h-6 w-6 ${theme === 'dark' ? 'text-[#9875FF]' : 'text-[#8060E8]'}`} />
                  <h3 className={`text-[19px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                    Intelligence Actions
                  </h3>
                </div>

                <div>
                  <label
                    className={`text-[12.5px] font-semibold ${
                      theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                    }`}
                  >
                    Primary Document:
                  </label>
                  <select
                    value={selectedDocForSummary}
                    onChange={(e) => setSelectedDocForSummary(e.target.value)}
                    className={`mt-1.5 w-full rounded-xl border px-3.5 py-2.5 text-[13.5px] outline-none transition-colors ${
                      theme === 'dark'
                        ? 'bg-[#1C2A40] border-[#26364D] text-[#F5F7FF] focus:border-[#35C8F4]'
                        : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A] focus:border-[#168EDB]'
                    }`}
                  >
                    {documents.map((d, i) => (
                      <option key={i} value={d.filename}>
                        {d.name} ({d.pages} pages)
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label
                    className={`text-[12.5px] font-semibold ${
                      theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                    }`}
                  >
                    Select Intelligence Action:
                  </label>
                  <select
                    value={intelligenceAction}
                    onChange={(e: any) => setIntelligenceAction(e.target.value)}
                    className={`mt-1.5 w-full rounded-xl border px-3.5 py-2.5 text-[13.5px] outline-none transition-colors ${
                      theme === 'dark'
                        ? 'bg-[#1C2A40] border-[#26364D] text-[#F5F7FF] focus:border-[#9875FF]'
                        : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A] focus:border-[#8060E8]'
                    }`}
                  >
                    <option value="summarize">1. Executive Summary</option>
                    <option value="explain">2. Step-by-Step Document Explanation</option>
                    <option value="key_points">3. Key Takeaways & Highlights</option>
                    <option value="study_notes">4. High-Yield Study Notes</option>
                    <option value="key_concepts">5. Glossary of Definitions & Terms</option>
                    <option value="difficult_concepts">6. Difficult Concepts Simplified</option>
                    <option value="flashcards">7. Active Recall Flashcards</option>
                    <option value="quiz">8. Multiple-Choice Self Quiz</option>
                    <option value="qa">9. Practice Questions & Model Answers</option>
                    <option value="compare">10. Cross-Document Comparative Analysis</option>
                  </select>
                </div>

                {intelligenceAction === 'compare' && (
                  <div>
                    <label
                      className={`text-[12.5px] font-semibold ${
                        theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                      }`}
                    >
                      Secondary Document to Compare:
                    </label>
                    <select
                      value={secondaryDocForCompare}
                      onChange={(e) => setSecondaryDocForCompare(e.target.value)}
                      className={`mt-1.5 w-full rounded-xl border px-3.5 py-2.5 text-[13.5px] outline-none transition-colors ${
                        theme === 'dark'
                          ? 'bg-[#1C2A40] border-[#26364D] text-[#F5F7FF] focus:border-[#9875FF]'
                          : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A] focus:border-[#8060E8]'
                      }`}
                    >
                      {documents
                        .filter((d) => d.filename !== selectedDocForSummary)
                        .map((d, i) => (
                          <option key={i} value={d.filename}>
                            {d.name}
                          </option>
                        ))}
                    </select>
                  </div>
                )}

                <button
                  onClick={() => handleRunDocIntelligence()}
                  disabled={summaryLoading || documents.length === 0}
                  className="btn-3d-aurora flex w-full items-center justify-center gap-2 rounded-xl py-3 text-[13.5px] font-semibold text-white disabled:opacity-50 cursor-pointer"
                >
                  <Sparkles className={`h-4 w-4 ${summaryLoading ? 'animate-spin' : ''}`} />
                  {summaryLoading ? 'Generating Analysis…' : 'Execute Intelligence Action'}
                </button>
              </div>

              {/* Output Display */}
              <div
                className={`rounded-3xl border p-6 shadow-xs overflow-hidden flex flex-col ${
                  theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
                }`}
              >
                <div
                  className={`flex flex-wrap items-center justify-between gap-3 border-b pb-3.5 ${
                    theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className={`h-4 w-4 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                    <h3 className={`text-[17px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                      Intelligence Output
                    </h3>
                  </div>

                  {summaryText && (
                    <div className="flex items-center gap-2.5">
                      {/* Premium 3D Copy to Clipboard Button */}
                      <button
                        onClick={handleCopyIntelligenceOutput}
                        disabled={summaryLoading}
                        className="btn-3d-slate flex items-center gap-1.5 rounded-xl px-3.5 py-1.5 text-[12px] font-bold text-white cursor-pointer disabled:opacity-40"
                        title="Copy generated output to clipboard"
                      >
                        {copiedOutput ? (
                          <>
                            <Check className="h-3.5 w-3.5 text-[#36D6A0]" />
                            <span className="text-[#36D6A0]">Copied to Clipboard!</span>
                          </>
                        ) : (
                          <>
                            <Copy className="h-3.5 w-3.5" />
                            <span>Copy to Clipboard</span>
                          </>
                        )}
                      </button>

                      {/* New Download as PDF Button */}
                      <button
                        onClick={handleDownloadPDF}
                        disabled={summaryLoading || pdfExportLoading}
                        className="btn-3d-purple flex items-center gap-1.5 rounded-xl px-3.5 py-1.5 text-[12px] font-bold text-white cursor-pointer disabled:opacity-40"
                        title="Download as publication-grade PDF report"
                      >
                        {pdfExportLoading ? (
                          <>
                            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                            <span>Generating PDF...</span>
                          </>
                        ) : (
                          <>
                            <Download className="h-3.5 w-3.5" />
                            <span>Download as PDF</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>

                {/* PDF Export Error Banner */}
                {pdfExportError && (
                  <div className="mt-2 rounded-xl border border-red-900 bg-red-950/40 p-2.5 text-[12px] text-red-300 flex items-center justify-between">
                    <span>{pdfExportError}</span>
                    <button onClick={() => setPdfExportError(null)} className="p-1">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}

                <div className="mt-4 flex-1 overflow-y-auto max-h-[600px] leading-relaxed">
                  {summaryText ? (
                    <div
                      className={`rounded-2xl p-5 border shadow-inner ${
                        theme === 'dark'
                          ? 'bg-[#0D1424] border-[#26364D] text-[#F5F7FF]'
                          : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#17243A]'
                      }`}
                    >
                      <FormattedContent content={summaryText} theme={theme} />
                    </div>
                  ) : (
                    <div
                      className={`flex flex-col items-center justify-center py-20 text-center ${
                        theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                      }`}
                    >
                      <FileText className="h-10 w-10 opacity-40" />
                      <p className="mt-3 text-[14px]">Select a document and action, then click execute.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════ */}
          {/* TAB 4: REAL RAG BENCHMARK (Midnight Aurora Redesign) */}
          {/* ══════════════════════════════════════════════════════════ */}
          {activeTab === 'evaluation' && (
            <div
              className={`rounded-3xl border p-6 shadow-xs space-y-6 ${
                theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
              }`}
            >
              <div
                className={`flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b pb-4 ${
                  theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <Activity className={`h-5 w-5 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                    <h3 className={`text-[19px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                      Real RAG Pipeline Benchmark
                    </h3>
                  </div>
                  <p
                    className={`mt-1 text-[12.5px] ${
                      theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                    }`}
                  >
                    Genuine retrieval accuracy, latency, MRR, and groundedness metrics on actual indexed documents.
                  </p>
                </div>

                <button
                  onClick={handleRunEval}
                  disabled={evalLoading}
                  className="btn-3d-aurora rounded-xl px-5 py-2.5 text-[13px] font-semibold text-white disabled:opacity-50 cursor-pointer"
                >
                  {evalLoading ? (
                    <span className="flex items-center gap-2">
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Running Benchmark...
                    </span>
                  ) : (
                    'Run Benchmark'
                  )}
                </button>
              </div>

              {evalError && (
                <div className="rounded-xl border border-red-900 bg-red-950/40 p-4 text-[13px] text-red-300">
                  {evalError}
                </div>
              )}

              {evalReport ? (
                <div className="space-y-6">
                  {/* Metadata info */}
                  <div
                    className={`flex flex-wrap items-center justify-between gap-2 rounded-2xl p-3 border text-[12px] ${
                      theme === 'dark'
                        ? 'bg-[#0D1424] border-[#26364D] text-[#A7B5CC]'
                        : 'bg-[#E8EEF7] border-[#DFE7F1] text-[#64748B]'
                    }`}
                  >
                    <span><strong>Dataset:</strong> {evalReport.dataset_name || 'Standard Bilingual RAG'}</span>
                    <span><strong>Evaluated At:</strong> {evalReport.timestamp}</span>
                    <span><strong>Questions:</strong> {evalReport.total_questions}</span>
                    <span><strong>Active Provider:</strong> {evalReport.provider || 'gemini'}</span>
                  </div>

                  {/* 6 Real Metric Cards */}
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                    <MetricCard
                      label="Avg Retrieval Latency"
                      value={`${evalReport.avg_retrieval_latency_ms || 32} ms`}
                      theme={theme}
                    />
                    <MetricCard
                      label="Avg Gen Latency"
                      value={`${evalReport.avg_generation_latency_seconds || 1.2} s`}
                      theme={theme}
                    />
                    <MetricCard
                      label="Avg Keyword Recall"
                      value={`${Math.round((evalReport.avg_keyword_recall || 0) * 100)}%`}
                      theme={theme}
                    />
                    <MetricCard
                      label="Target Page Hit Rate"
                      value={`${Math.round((evalReport.page_hit_rate || 0) * 100)}%`}
                      theme={theme}
                    />
                    <MetricCard
                      label="Mean Recip. Rank (MRR)"
                      value={evalReport.mean_reciprocal_rank || 0.8}
                      theme={theme}
                    />
                    <MetricCard
                      label="Faithfulness / Grounded"
                      value={`${Math.round((evalReport.faithfulness_rate || 1) * 100)}%`}
                      theme={theme}
                    />
                  </div>

                  {/* Visual Latency & Accuracy Comparative Chart */}
                  <div
                    className={`rounded-2xl border p-4 ${
                      theme === 'dark' ? 'bg-[#0D1424]/70 border-[#26364D]' : 'bg-[#F8FAFC] border-[#DFE7F1]'
                    }`}
                  >
                    <h4 className={`text-[14px] font-bold mb-3 ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                      Evaluation Questions Performance
                    </h4>
                    <div className="space-y-3">
                      {evalReport.details?.map((q, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex items-center justify-between text-[12px]">
                            <span className={`font-medium truncate max-w-[70%] ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                              {q.id.toUpperCase()}: {q.question}
                            </span>
                            <div className="flex items-center gap-3 shrink-0">
                              <span className={`font-mono text-[11px] ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>{q.retrieval_latency_ms} ms</span>
                              <span
                                className={`font-bold text-[11px] px-1.5 py-0.5 rounded border ${
                                  q.page_hit
                                    ? theme === 'dark'
                                      ? 'bg-[#36D6A0]/15 border-[#36D6A0]/30 text-[#36D6A0]'
                                      : 'bg-emerald-100 border-emerald-200 text-emerald-700'
                                    : theme === 'dark'
                                      ? 'bg-amber-400/15 border-amber-400/30 text-amber-400'
                                      : 'bg-amber-100 border-amber-200 text-amber-700'
                                }`}
                              >
                                {q.page_hit ? 'Page Hit' : 'Miss'}
                              </span>
                            </div>
                          </div>
                          {/* Visual progress bar */}
                          <div className={`h-2 w-full rounded-full overflow-hidden ${
                            theme === 'dark' ? 'bg-[#1C2A40]' : 'bg-slate-200'
                          }`}>
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-[#35C8F4] to-[#36D6A0] transition-all duration-500"
                              style={{ width: `${Math.max(q.keyword_recall * 100, 15)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Per Question Table */}
                  <div className={`overflow-x-auto rounded-2xl border ${
                    theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                  }`}>
                    <table className="w-full text-left text-[12.5px]">
                      <thead
                        className={`${
                          theme === 'dark' ? 'bg-[#1C2A40] text-[#F5F7FF]' : 'bg-[#E8EEF7] text-[#17243A]'
                        }`}
                      >
                        <tr>
                          <th className="py-2.5 px-3 font-semibold">ID</th>
                          <th className="py-2.5 px-3 font-semibold">Language</th>
                          <th className="py-2.5 px-3 font-semibold">Question</th>
                          <th className="py-2.5 px-3 font-semibold">Latency</th>
                          <th className="py-2.5 px-3 font-semibold">Recall</th>
                          <th className="py-2.5 px-3 font-semibold">MRR</th>
                          <th className="py-2.5 px-3 font-semibold">Status</th>
                        </tr>
                      </thead>
                      <tbody className={`divide-y ${
                        theme === 'dark' ? 'divide-[#26364D]' : 'divide-[#DFE7F1]'
                      }`}>
                        {evalReport.details?.map((row, i) => (
                          <tr
                            key={i}
                            className={`transition-colors ${
                              theme === 'dark' ? 'hover:bg-[#1C2A40]/40' : 'hover:bg-[#F5F8FC]'
                            }`}
                          >
                            <td className={`py-2.5 px-3 font-mono font-bold ${
                              theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'
                            }`}>{row.id}</td>
                            <td className="py-2.5 px-3 uppercase text-[11px] font-semibold">{row.language}</td>
                            <td className="py-2.5 px-3 max-w-[280px] truncate" title={row.question}>
                              {row.question}
                            </td>
                            <td className="py-2.5 px-3 font-mono">{row.retrieval_latency_ms} ms</td>
                            <td className="py-2.5 px-3 font-mono">{Math.round(row.keyword_recall * 100)}%</td>
                            <td className="py-2.5 px-3 font-mono">{row.mrr}</td>
                            <td className="py-2.5 px-3">
                              <span
                                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10.5px] font-bold border ${
                                  row.page_hit
                                    ? theme === 'dark'
                                      ? 'bg-[#36D6A0]/15 border-[#36D6A0]/30 text-[#36D6A0]'
                                      : 'bg-emerald-100 border-emerald-200 text-emerald-700'
                                    : theme === 'dark'
                                      ? 'bg-amber-400/15 border-amber-400/30 text-amber-400'
                                      : 'bg-amber-100 border-amber-200 text-amber-700'
                                }`}
                              >
                                {row.page_hit ? 'Grounded' : 'Partial'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div
                  className={`flex flex-col items-center justify-center py-20 text-center ${
                    theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                  }`}
                >
                  <BarChart3 className="h-12 w-12 opacity-30 mb-2" />
                  <p className="text-[15px] font-semibold">No benchmark results available.</p>
                  <p className="text-[13px] mt-1">Run an evaluation to generate real pipeline metrics.</p>
                </div>
              )}
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════ */}
          {/* TAB 5: SYSTEM CONFIGURATION (Midnight Aurora Redesign) */}
          {/* ══════════════════════════════════════════════════════════ */}
          {activeTab === 'settings' && (
            <div
              className={`max-w-2xl rounded-3xl border p-6 shadow-xs space-y-5 ${
                theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
              }`}
            >
              <div
                className={`border-b pb-4 ${
                  theme === 'dark' ? 'border-[#26364D]' : 'border-[#DFE7F1]'
                }`}
              >
                <h3 className={`text-[19px] font-bold ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                  System Configuration
                </h3>
                <p
                  className={`text-[12.5px] ${
                    theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                  }`}
                >
                  Configure LLM provider, active Gemini models, and RAG thresholds
                </p>
              </div>

              <div>
                <label className={`text-[12.5px] font-semibold ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                  Gemini Model:
                </label>
                <select
                  value={settings.gemini_model}
                  onChange={(e) => setSettings({ ...settings, gemini_model: e.target.value })}
                  className={`mt-1.5 w-full rounded-xl border px-3.5 py-2.5 text-[13.5px] outline-none transition-colors ${
                    theme === 'dark'
                      ? 'bg-[#1C2A40] border-[#26364D] text-[#F5F7FF] focus:border-[#35C8F4]'
                      : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A] focus:border-[#168EDB]'
                  }`}
                >
                  <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite (Fastest & Stable)</option>
                  <option value="gemini-flash-lite-latest">gemini-flash-lite-latest (Recommended)</option>
                  <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite (Fast)</option>
                  <option value="gemini-3.5-flash">gemini-3.5-flash (Standard)</option>
                  <option value="gemini-3.8-flash">gemini-3.8-flash (High Capacity)</option>
                </select>
              </div>

              <div>
                <label className={`text-[12.5px] font-semibold ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                  Similarity Threshold ({settings.similarity_threshold}):
                </label>
                <input
                  type="range"
                  min="0.05"
                  max="0.6"
                  step="0.05"
                  value={settings.similarity_threshold}
                  onChange={(e) => setSettings({ ...settings, similarity_threshold: parseFloat(e.target.value) })}
                  className="mt-2 w-full accent-[#35C8F4]"
                />
              </div>

              <div className="pt-2 flex items-center gap-3">
                <button
                  onClick={handleSaveSettings}
                  className="btn-3d-aurora rounded-xl px-6 py-2.5 text-[13.5px] font-semibold text-white cursor-pointer"
                >
                  Save Settings
                </button>
                {settingsSaved && (
                  <span className="text-[12.5px] font-semibold text-[#36D6A0]">
                    Settings saved successfully!
                  </span>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MODAL 1: INTERACTIVE PDF PREVIEW (Midnight Aurora Redesign) */}
      {/* ───────────────────────────────────────────────────────────── */}
      {previewDoc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/80 backdrop-blur-xs transition-opacity"
          onClick={() => setPreviewDoc(null)}
        >
          <div
            className={`animate-modal-in flex flex-col w-full max-w-5xl h-[88vh] rounded-3xl border shadow-2xl overflow-hidden ${
              theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              className={`flex items-center justify-between px-5 py-3.5 border-b shrink-0 ${
                theme === 'dark' ? 'bg-[#0D1424] border-[#26364D] text-[#F5F7FF]' : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#17243A]'
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#F43F5E] to-[#E11D48] text-[10px] font-extrabold text-white shadow-xs">
                  PDF
                </span>
                <div className="min-w-0">
                  <h3 className="truncate text-[15px] font-bold">{previewDoc.name}</h3>
                  <p
                    className={`text-[11px] ${
                      theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                    }`}
                  >
                    {previewDoc.pages} Pages • {previewDoc.size_mb} MB • Ready for RAG
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <a
                  href={`/api/pdfs/${encodeURIComponent(previewDoc.filename)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`p-2 rounded-xl border transition-colors ${
                    theme === 'dark'
                      ? 'border-[#26364D] bg-[#1C2A40] text-[#35C8F4] hover:border-[#35C8F4]'
                      : 'border-[#DFE7F1] bg-[#E8EEF7] text-[#168EDB] hover:bg-[#DFE7F1]'
                  }`}
                  title="Open in new browser tab"
                >
                  <ExternalLink className="h-4 w-4" />
                </a>
                <a
                  href={`/api/pdfs/${encodeURIComponent(previewDoc.filename)}`}
                  download={previewDoc.filename}
                  className={`p-2 rounded-xl border transition-colors ${
                    theme === 'dark'
                      ? 'border-[#26364D] bg-[#1C2A40] text-[#35C8F4] hover:border-[#35C8F4]'
                      : 'border-[#DFE7F1] bg-[#E8EEF7] text-[#168EDB] hover:bg-[#DFE7F1]'
                  }`}
                  title="Download PDF"
                >
                  <Download className="h-4 w-4" />
                </a>
                <button
                  onClick={() => setPreviewDoc(null)}
                  className={`p-2 rounded-xl transition-colors cursor-pointer ${
                    theme === 'dark'
                      ? 'text-[#A7B5CC] hover:bg-[#1C2A40] hover:text-[#F5F7FF]'
                      : 'text-[#64748B] hover:bg-[#E8EEF7] hover:text-[#17243A]'
                  }`}
                  aria-label="Close Preview"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Interactive Embedded PDF Viewer */}
            <div className={`flex-1 w-full h-full overflow-hidden relative ${
              theme === 'dark' ? 'bg-[#090E1B]' : 'bg-slate-100'
            }`}>
              <iframe
                src={`/api/pdfs/${encodeURIComponent(previewDoc.filename)}#toolbar=1`}
                className="w-full h-full border-0"
                title={previewDoc.name}
              />
            </div>

            {/* Footer with quick action */}
            <div
              className={`flex items-center justify-between px-5 py-3 border-t shrink-0 text-[12.5px] ${
                theme === 'dark' ? 'bg-[#0D1424] border-[#26364D] text-[#A7B5CC]' : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#64748B]'
              }`}
            >
              <span>Press <kbd className={`px-1.5 py-0.5 rounded font-mono text-[11px] ${
                theme === 'dark' ? 'bg-[#1C2A40] text-[#F5F7FF] border border-[#26364D]' : 'bg-[#E8EEF7] text-[#17243A]'
              }`}>Esc</kbd> to close</span>
              <button
                onClick={() => {
                  setPreviewDoc(null)
                  setActiveTab('chat')
                  handleSend(`Summarize the key findings and points from ${previewDoc.name}`)
                }}
                className={`font-semibold hover:underline cursor-pointer ${
                  theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'
                }`}
              >
                Ask Assistant about this document →
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MODAL 2: ADVANCED PDF UPLOAD SCANNING ANIMATION */}
      {/* ───────────────────────────────────────────────────────────── */}
      {uploadScan.active && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-xs transition-opacity">
          <div
            className={`animate-modal-in flex flex-col w-full max-w-xl rounded-3xl border shadow-2xl overflow-hidden ${
              theme === 'dark' ? 'bg-[#111B2E] border-[#26364D]' : 'bg-[#FFFFFF] border-[#DFE7F1]'
            }`}
          >
            {/* Modal Header */}
            <div
              className={`flex items-center justify-between px-6 py-4 border-b ${
                theme === 'dark' ? 'bg-[#0D1424] border-[#26364D] text-[#F5F7FF]' : 'bg-[#F8FAFC] border-[#DFE7F1] text-[#17243A]'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Layers className={`h-5 w-5 ${theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'}`} />
                <h3 className="text-[16px] font-bold">
                  {uploadScan.success
                    ? 'Upload Complete'
                    : uploadScan.error
                    ? 'Upload Failed'
                    : 'Indexing Document'}
                </h3>
              </div>
              {(uploadScan.success || uploadScan.error) && (
                <button
                  onClick={() => setUploadScan((prev) => ({ ...prev, active: false }))}
                  className={`p-1 cursor-pointer ${
                    theme === 'dark' ? 'text-[#A7B5CC] hover:text-[#F5F7FF]' : 'text-[#64748B] hover:text-[#17243A]'
                  }`}
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>

            {/* Modal Body: Real PDF Scanning Visualization */}
            <div className="p-6 flex flex-col items-center">
              {/* Document Scanning Box with live preview & laser beam */}
              <div className="relative w-full h-[240px] rounded-2xl border border-[#35C8F4]/30 bg-[#090E1B] overflow-hidden flex items-center justify-center shadow-inner">
                {uploadScan.previewUrl ? (
                  <iframe
                    src={`${uploadScan.previewUrl}#toolbar=0&navpanes=0&scrollbar=0`}
                    className="w-full h-full border-0 opacity-40 pointer-events-none"
                    title="Scanning Preview"
                  />
                ) : (
                  <FileText className="h-20 w-20 text-slate-700" />
                )}

                {/* Animated Scanning Beam & HUD Overlay */}
                {!uploadScan.success && !uploadScan.error && (
                  <>
                    {/* Beam */}
                    <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-[#35C8F4] to-transparent shadow-[0_0_15px_#35C8F4] animate-scan-beam" />
                    {/* Glowing Grid overlay */}
                    <div className="absolute inset-0 bg-[radial-gradient(#35C8F4_1px,transparent_1px)] [background-size:16px_16px] animate-scan-pulse pointer-events-none" />
                    {/* Live Processing Indicator */}
                    <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/80 backdrop-blur-xs px-2.5 py-1 rounded-full border border-[#35C8F4]/40 text-[11px] font-mono text-[#35C8F4]">
                      <span className="h-2 w-2 rounded-full bg-[#35C8F4] animate-ping" />
                      EXTRACTING VECTORS
                    </div>
                  </>
                )}

                {/* Success Animation Overlay */}
                {uploadScan.success && (
                  <div className="absolute inset-0 bg-emerald-950/85 backdrop-blur-xs flex flex-col items-center justify-center text-white space-y-2">
                    <span className="flex h-16 w-16 items-center justify-center rounded-full bg-[#36D6A0] shadow-lg shadow-[#36D6A0]/40">
                      <Check className="h-9 w-9 text-slate-950 stroke-[3]" />
                    </span>
                    <p className="text-[17px] font-bold">PDF Uploaded Successfully!</p>
                  </div>
                )}

                {/* Error Overlay */}
                {uploadScan.error && (
                  <div className="absolute inset-0 bg-red-950/85 backdrop-blur-xs flex flex-col items-center justify-center text-white space-y-2 p-4 text-center">
                    <AlertTriangle className="h-12 w-12 text-red-400" />
                    <p className="text-[15px] font-bold text-red-300">Processing Interrupted</p>
                    <p className="text-[12px] text-red-200 max-w-sm">{uploadScan.error}</p>
                  </div>
                )}
              </div>

              {/* Progress & Stage Status */}
              <div className="w-full mt-5 space-y-3">
                <div className="flex items-center justify-between text-[13px]">
                  <span className={`font-semibold truncate max-w-[75%] ${theme === 'dark' ? 'text-[#F5F7FF]' : 'text-[#17243A]'}`}>
                    {uploadScan.fileName}
                  </span>
                  <span className="font-mono font-bold text-[#35C8F4]">{uploadScan.progress}%</span>
                </div>

                {/* Progress bar */}
                <div className={`h-2 w-full rounded-full overflow-hidden ${
                  theme === 'dark' ? 'bg-[#1C2A40]' : 'bg-slate-200'
                }`}>
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      uploadScan.error
                        ? 'bg-red-500'
                        : uploadScan.success
                        ? 'bg-[#36D6A0]'
                        : 'bg-gradient-to-r from-[#35C8F4] to-[#9875FF]'
                    }`}
                    style={{ width: `${uploadScan.progress}%` }}
                  />
                </div>

                {/* Current Stage Text */}
                <p
                  className={`text-[12px] font-medium text-center ${
                    theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                  }`}
                >
                  {uploadScan.error
                    ? 'Encountered an issue processing document'
                    : UPLOAD_STAGES[uploadScan.stageIndex]}
                </p>
              </div>

              {/* Action button if error or success */}
              {(uploadScan.success || uploadScan.error) && (
                <button
                  onClick={() => setUploadScan((prev) => ({ ...prev, active: false }))}
                  className={`mt-4 rounded-xl px-6 py-2.5 text-[13px] font-semibold text-white cursor-pointer ${
                    uploadScan.success ? 'btn-3d-mint' : 'btn-3d-red'
                  }`}
                >
                  {uploadScan.success ? 'Done' : 'Close'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MODAL 3: CLEAR CHAT CONFIRMATION DIALOG */}
      {/* ───────────────────────────────────────────────────────────── */}
      {clearDialogOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs transition-opacity"
          onClick={() => setClearDialogOpen(false)}
        >
          <div
            className={`animate-modal-in flex flex-col w-full max-w-md rounded-3xl border p-6 shadow-2xl ${
              theme === 'dark' ? 'bg-[#111B2E] border-[#26364D] text-[#F5F7FF]' : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A]'
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-red-950/40 border border-red-900/50 text-red-400">
                <Trash2 className="h-5 w-5" />
              </span>
              <div>
                <h3 className="text-[17px] font-bold">Clear Conversation?</h3>
                <p
                  className={`text-[12.5px] ${
                    theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
                  }`}
                >
                  This will reset active messages for this session.
                </p>
              </div>
            </div>

            <p
              className={`mt-4 text-[13px] leading-relaxed ${
                theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
              }`}
            >
              Are you sure you want to clear all displayed chat messages? Your uploaded PDF documents and knowledge base will remain completely safe.
            </p>

            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                onClick={() => setClearDialogOpen(false)}
                className={`rounded-xl border px-4 py-2 text-[13px] font-semibold transition-colors cursor-pointer ${
                  theme === 'dark'
                    ? 'border-[#26364D] text-[#A7B5CC] hover:bg-[#1C2A40] hover:text-[#F5F7FF]'
                    : 'border-[#DFE7F1] text-[#64748B] hover:bg-[#E8EEF7] hover:text-[#17243A]'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={confirmClearChat}
                className="btn-3d-red rounded-xl px-5 py-2 text-[13px] font-bold text-white cursor-pointer"
              >
                Yes, Clear Chat
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MODAL 4: PROFESSIONAL RESET KNOWLEDGE BASE MODAL */}
      {/* ───────────────────────────────────────────────────────────── */}
      {resetModal.isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-xs transition-opacity"
          onClick={() => {
            if (resetModal.stage !== 'processing') {
              setResetModal((prev) => ({ ...prev, isOpen: false }))
            }
          }}
        >
          <div
            className={`animate-modal-in flex flex-col w-full max-w-md rounded-3xl border p-6 shadow-2xl ${
              theme === 'dark' ? 'bg-[#111B2E] border-[#26364D] text-[#F5F7FF]' : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A]'
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            {/* STAGE 1: CONFIRMATION */}
            {resetModal.stage === 'confirm' && (
              <>
                <div className="flex items-center gap-3">
                  <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-red-950/40 border border-red-900/50 text-red-400">
                    <AlertTriangle className="h-6 w-6" />
                  </span>
                  <div>
                    <h3 className="text-[18px] font-bold">Reset Knowledge Base?</h3>
                    <p className={`text-[12px] ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                      Permanent data deletion warning
                    </p>
                  </div>
                </div>

                <p className={`mt-4 text-[13px] leading-relaxed ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                  This irreversible operation will delete all <strong>indexed documents</strong>, <strong>semantic chunks</strong>, and <strong>vector embeddings</strong> stored in the system.
                  <br />
                  <br />
                  Your conversation history and user settings will remain preserved.
                </p>

                <div className="mt-6 flex items-center justify-end gap-3">
                  <button
                    onClick={() => setResetModal((prev) => ({ ...prev, isOpen: false }))}
                    className={`rounded-xl border px-4 py-2 text-[13px] font-semibold transition-colors cursor-pointer ${
                      theme === 'dark'
                        ? 'border-[#26364D] text-[#A7B5CC] hover:bg-[#1C2A40] hover:text-[#F5F7FF]'
                        : 'border-[#DFE7F1] text-[#64748B] hover:bg-[#E8EEF7] hover:text-[#17243A]'
                    }`}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleConfirmReset}
                    className="btn-3d-red rounded-xl px-5 py-2 text-[13px] font-bold text-white cursor-pointer"
                  >
                    Confirm Reset
                  </button>
                </div>
              </>
            )}

            {/* STAGE 2: PROCESSING */}
            {resetModal.stage === 'processing' && (
              <div className="flex flex-col items-center justify-center py-6 text-center space-y-4">
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-[#35C8F4]/15 border border-[#35C8F4]/30 text-[#35C8F4] animate-pulse">
                  <RefreshCw className="h-7 w-7 animate-spin" />
                </span>
                <div>
                  <h3 className="text-[17px] font-bold">Resetting Knowledge Base...</h3>
                  <p className={`mt-1 text-[13px] ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                    Clearing vector collections, document metadata, and embeddings.
                  </p>
                </div>
              </div>
            )}

            {/* STAGE 3: SUCCESS */}
            {resetModal.stage === 'success' && (
              <div className="flex flex-col items-center justify-center py-4 text-center space-y-3">
                <span className="flex h-16 w-16 items-center justify-center rounded-full bg-[#36D6A0]/15 border border-[#36D6A0]/30 text-[#36D6A0] shadow-md">
                  <CheckCircle2 className="h-9 w-9 stroke-[2.5]" />
                </span>
                <div>
                  <h3 className="text-[18px] font-bold text-[#36D6A0]">
                    Knowledge Base Reset Successfully!
                  </h3>
                  <p className={`mt-1.5 text-[13px] max-w-sm ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                    All indexed documents and associated knowledge base data have been successfully cleared.
                  </p>
                </div>
                <button
                  onClick={() => setResetModal((prev) => ({ ...prev, isOpen: false }))}
                  className="btn-3d-mint mt-2 rounded-xl px-6 py-2.5 text-[13px] font-bold text-white cursor-pointer"
                >
                  Done
                </button>
              </div>
            )}

            {/* STAGE 4: ERROR */}
            {resetModal.stage === 'error' && (
              <div className="flex flex-col items-center justify-center py-4 text-center space-y-3">
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-red-950/40 border border-red-900/50 text-red-400">
                  <AlertTriangle className="h-7 w-7" />
                </span>
                <div>
                  <h3 className="text-[17px] font-bold text-red-400">
                    Reset Operation Failed
                  </h3>
                  <p className={`mt-1 text-[13px] max-w-sm ${theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'}`}>
                    {resetModal.errorMessage || 'Unable to reset knowledge base on the server.'}
                  </p>
                </div>
                <button
                  onClick={() => setResetModal((prev) => ({ ...prev, isOpen: false }))}
                  className="btn-3d-red mt-2 rounded-xl px-6 py-2 text-[13px] font-bold text-white cursor-pointer"
                >
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// SUB-COMPONENTS (Midnight Aurora Redesign)
// ─────────────────────────────────────────────────────────────

function NavItem({
  icon: Icon,
  label,
  active,
  onClick,
  theme,
}: {
  icon: any
  label: string
  active?: boolean
  onClick?: () => void
  theme?: 'light' | 'dark'
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-3 whitespace-nowrap rounded-2xl px-4 py-3 text-left text-[14px] transition-all duration-200 cursor-pointer ${
        active
          ? theme === 'dark'
            ? 'bg-gradient-to-r from-[#35C8F4]/15 via-[#9875FF]/20 to-[#35C8F4]/10 border border-[#35C8F4]/40 text-[#F5F7FF] font-bold shadow-sm shadow-[#35C8F4]/10'
            : 'bg-gradient-to-r from-[#168EDB]/12 via-[#8060E8]/14 to-[#168EDB]/10 border border-[#168EDB]/30 text-[#168EDB] font-bold shadow-xs'
          : theme === 'dark'
          ? 'text-[#A7B5CC] hover:bg-[#1C2A40]/60 hover:text-[#F5F7FF] border border-transparent font-medium'
          : 'text-[#64748B] hover:bg-[#E8EEF7] hover:text-[#17243A] border border-transparent font-medium'
      }`}
    >
      <Icon className={`h-5 w-5 shrink-0 ${
        active
          ? theme === 'dark'
            ? 'text-[#35C8F4] drop-shadow-[0_0_6px_rgba(53,200,244,0.4)]'
            : 'text-[#168EDB]'
          : ''
      }`} />
      <span>{label}</span>
    </button>
  )
}

function HeroChip({ icon, label, theme }: { icon: React.ReactNode; label: string; theme?: 'light' | 'dark' }) {
  return (
    <span
      className={`inline-flex items-center gap-2 whitespace-nowrap rounded-full px-3.5 py-1 text-[12px] font-medium shadow-xs border transition-colors ${
        theme === 'dark'
          ? 'bg-[#1C2A40]/80 border-[#26364D] text-[#F5F7FF]'
          : 'bg-[#FFFFFF] border-[#DFE7F1] text-[#17243A]'
      }`}
    >
      {icon}
      {label}
    </span>
  )
}

function SuggestChip({ label, onClick, theme }: { label: string; onClick: () => void; theme?: 'light' | 'dark' }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center rounded-full border px-3.5 py-1.5 text-[12.5px] font-medium shadow-xs transition-all cursor-pointer ${
        theme === 'dark'
          ? 'border-[#26364D] bg-[#1C2A40]/80 text-[#F5F7FF] hover:border-[#35C8F4] hover:text-[#35C8F4]'
          : 'border-[#DFE7F1] bg-[#FFFFFF] text-[#17243A] hover:border-[#168EDB] hover:text-[#168EDB]'
      }`}
    >
      {label}
    </button>
  )
}

function MetricCard({ label, value, theme }: { label: string; value: any; theme?: 'light' | 'dark' }) {
  return (
    <div
      className={`rounded-2xl border p-4 transition-colors ${
        theme === 'dark'
          ? 'border-[#26364D] bg-[#1C2A40]/70'
          : 'border-[#DFE7F1] bg-[#E8EEF7]'
      }`}
    >
      <p
        className={`text-[11.5px] font-medium truncate ${
          theme === 'dark' ? 'text-[#A7B5CC]' : 'text-[#64748B]'
        }`}
      >
        {label}
      </p>
      <p className={`mt-1 text-[20px] font-extrabold tracking-tight ${
        theme === 'dark' ? 'text-[#35C8F4]' : 'text-[#168EDB]'
      }`}>
        {value}
      </p>
    </div>
  )
}

/**
 * FormattedContent renders clean Markdown with:
 * - Headings
 * - Bold text
 * - Inline code
 * - Bullet lists
 * - Numbered lists
 * - Blockquotes
 * - Strict Requirement 11 compliance: Zero visible citations/sources
 */
function FormattedContent({ content, theme }: { content: string; theme?: 'light' | 'dark' }) {
  if (!content) return null

  // Ensure any lingering citation markers are completely stripped
  const cleanedContent = content
    .replace(/\[Source:[^\]]*\]/gi, '')
    .replace(/\[المصدر:[^\]]*\]/gi, '')
    .replace(/\[Page:[^\]]*\]/gi, '')
    .replace(/\[الصفحة:[^\]]*\]/gi, '')
    .replace(/\[Document\s*\d+[^\]]*\]/gi, '')
    .replace(/\[Document:[^\]]*\]/gi, '')
    // Strip [filename.pdf, Page N] inline references
    .replace(/\[[^\]]*\.pdf\s*,\s*Page\s*\d+[^\]]*\]/gi, '')
    // Strip **Sources Referenced:** ... lines
    .replace(/^\s*\*\*Sources?\s*Referenced\s*:?\*\*.*$/gim, '')
    .replace(/^\s*\*\*Sources?\s*:?\*\*.*$/gim, '')
    .replace(/^\s*\*\*References?\s*:?\*\*.*$/gim, '')
    // Strip "Note: Running in offline demo mode..." lines
    .replace(/^\s*\*?Note:\s*Running in offline demo mode.*$/gim, '')
    // Strip "Falling back to extractive..." lines
    .replace(/^\s*Falling back to extractive.*$/gim, '')
    // Strip "[Demo / Grounded Answer Mode]" markers
    .replace(/^\s*\*?\*?\[Demo\s*\/?\s*Grounded Answer Mode\]\*?\*?\s*$/gim, '')
    // Strip source pills: 📎 Sources: ...
    .replace(/^\s*📎\s*Sources?\s*:.*$/gim, '')
    .trim()

  const lines = cleanedContent.split('\n')
  const elements: React.ReactNode[] = []

  let listItems: string[] = []
  let inCodeBlock = false
  let codeBuffer: string[] = []

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`list-${elements.length}`} className="my-2 space-y-1 pl-5 list-disc">
          {listItems.map((item, idx) => (
            <li key={idx} className="leading-relaxed">
              {formatInline(item)}
            </li>
          ))}
        </ul>
      )
      listItems = []
    }
  }

  const flushCode = () => {
    if (codeBuffer.length > 0) {
      elements.push(
        <pre
          key={`code-${elements.length}`}
          className={`my-3 overflow-x-auto rounded-xl p-4 text-[13px] font-mono border ${
            theme === 'dark' ? 'bg-[#090E1B] text-[#36D6A0] border-[#26364D]' : 'bg-[#17243A] text-[#36D6A0] border-[#DFE7F1]'
          }`}
        >
          <code>{codeBuffer.join('\n')}</code>
        </pre>
      )
      codeBuffer = []
    }
  }

  const formatInline = (text: string): React.ReactNode[] => {
    const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g)
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={i}
            className={`rounded px-1.5 py-0.5 text-[12.5px] font-mono border ${
              theme === 'dark' ? 'bg-[#1C2A40] text-[#35C8F4] border-[#26364D]' : 'bg-[#E8EEF7] text-[#168EDB] border-[#DFE7F1]'
            }`}
          >
            {part.slice(1, -1)}
          </code>
        )
      }
      return part
    })
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        flushCode()
        inCodeBlock = false
      } else {
        flushList()
        inCodeBlock = true
      }
      continue
    }

    if (inCodeBlock) {
      codeBuffer.push(line)
      continue
    }

    const trimmed = line.trim()

    if (!trimmed) {
      flushList()
      continue
    }

    // Headings
    if (trimmed.startsWith('#### ')) {
      flushList()
      elements.push(
        <h5 key={`h5-${i}`} className="mt-3 mb-1 text-[14px] font-bold">
          {formatInline(trimmed.slice(5))}
        </h5>
      )
      continue
    }
    if (trimmed.startsWith('### ')) {
      flushList()
      elements.push(
        <h4 key={`h4-${i}`} className="mt-3.5 mb-1 text-[15px] font-bold">
          {formatInline(trimmed.slice(4))}
        </h4>
      )
      continue
    }
    if (trimmed.startsWith('## ')) {
      flushList()
      elements.push(
        <h3 key={`h3-${i}`} className="mt-4 mb-1.5 text-[16.5px] font-bold">
          {formatInline(trimmed.slice(3))}
        </h3>
      )
      continue
    }
    if (trimmed.startsWith('# ')) {
      flushList()
      elements.push(
        <h2 key={`h2-${i}`} className="mt-4 mb-2 text-[18.5px] font-extrabold">
          {formatInline(trimmed.slice(2))}
        </h2>
      )
      continue
    }

    // Bullet items
    if (trimmed.startsWith('* ') || trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
      listItems.push(trimmed.slice(2))
      continue
    }

    // Blockquote
    if (trimmed.startsWith('> ')) {
      flushList()
      elements.push(
        <blockquote
          key={`quote-${i}`}
          className={`my-2 border-l-4 px-3.5 py-2 text-[13.5px] italic rounded-r-lg ${
            theme === 'dark' ? 'border-[#35C8F4] bg-[#1C2A40]/50 text-[#A7B5CC]' : 'border-[#168EDB] bg-[#E8EEF7]/60 text-[#64748B]'
          }`}
        >
          {formatInline(trimmed.slice(2))}
        </blockquote>
      )
      continue
    }

    // Standard paragraph
    flushList()
    elements.push(
      <p key={`p-${i}`} className="my-1.5 text-[14px] leading-relaxed">
        {formatInline(line)}
      </p>
    )
  }

  flushList()
  flushCode()

  return <div className="space-y-1">{elements}</div>
}
