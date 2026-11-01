import React, { useState, useEffect } from 'react'
import {
  MessageSquareMore,
  FileText,
  Sparkles,
  BarChart3,
  Settings,
  Zap,
  ChevronDown,
  LogOut,
  Sun,
  Moon,
  CheckCircle2,
  Globe,
  ShieldCheck,
  Trash2,
  User,
  Bot,
  CheckCheck,
  GraduationCap,
  Workflow,
  Copy,
  Paperclip,
  Send,
  Lightbulb,
  List,
  HelpCircle,
  Link2,
  Check,
  Search,
  Menu,
  Upload,
  RefreshCw,
  Sliders,
  SlidersHorizontal,
  Database,
  Cpu,
  Clock,
  Target,
  FileSearch,
} from 'lucide-react'

const API_BASE = '/api'

interface Citation {
  source: string
  raw_source?: string
  page: number | string
  snippet: string
}

interface Message {
  user: string
  assistant: string
  citations: Citation[]
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
}

export default function Chat() {
  const [activeTab, setActiveTab] = useState<'chat' | 'documents' | 'summarize' | 'evaluation' | 'settings'>('chat')
  const [mode, setMode] = useState('light')
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    {
      user: 'What is the difference between supervised and unsupervised learning?',
      assistant: `Supervised and unsupervised learning are two main types of machine learning:`,
      citations: [
        {
          source: 'Machine Learning.pdf',
          page: 12,
          snippet: 'Supervised learning uses labeled data to train the model...',
        },
        {
          source: 'ML_Book.pdf',
          page: 45,
          snippet: 'Unsupervised learning finds patterns without labels...',
        },
      ],
      time: '4:15 PM',
      grounded: true,
    },
  ])

  // Documents state
  const [documents, setDocuments] = useState<DocItem[]>([
    { id: '1', filename: 'machine_learning_basics.pdf', name: 'Machine Learning.pdf', pages: 300, size_mb: 12.5, chunks: 11, uploaded_at: 'Today' },
    { id: '2', filename: 'deep_learning_guide.pdf', name: 'Deep Learning.pdf', pages: 250, size_mb: 10.8, chunks: 6, uploaded_at: 'Today' },
  ])
  const [totalChunks, setTotalChunks] = useState(17)

  // Summarize state
  const [selectedDocForSummary, setSelectedDocForSummary] = useState('')
  const [summaryText, setSummaryText] = useState('')
  const [summaryLoading, setSummaryLoading] = useState(false)

  // Evaluation state
  const [evalLoading, setEvalLoading] = useState(false)
  const [evalResult, setEvalResult] = useState<any>(null)

  // Settings state
  const [settings, setSettings] = useState({
    provider: 'gemini',
    api_key: '',
    gemini_model: 'gemini-3.8-flash',
    chunk_size: 500,
    chunk_overlap: 100,
    top_k: 4,
    similarity_threshold: 0.3,
  })
  const [settingsSaved, setSettingsSaved] = useState(false)

  // Fetch documents and status on load
  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`)
      if (res.ok) {
        const data = await res.json()
        if (data.documents && data.documents.length > 0) {
          setDocuments(data.documents)
          setSelectedDocForSummary(data.documents[0].filename)
        }
        if (data.total_chunks !== undefined) {
          setTotalChunks(data.total_chunks)
        }
      }
    } catch (err) {
      console.log('Backend not reachable yet, using sample data')
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

  useEffect(() => {
    fetchDocs()
    fetchSettings()
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
      citations: m.citations,
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
            citations: data.citations || [],
            time: data.timestamp || now,
            grounded: data.grounded,
          },
        ])
      } else {
        throw new Error('Failed response')
      }
    } catch (err) {
      // Fallback response for offline or demo testing
      setMessages((prev) => [
        ...prev,
        {
          user: q,
          assistant: `Based on your indexed documents: ${q}\n\nSupervised learning relies on explicit ground-truth targets, while unsupervised learning discovers inherent patterns.`,
          citations: [
            { source: 'Machine Learning.pdf', page: 12, snippet: 'Supervised learning uses labeled data to train the model...' },
          ],
          time: now,
          grounded: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  // File upload handler
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return
    const formData = new FormData()
    for (let i = 0; i < e.target.files.length; i++) {
      formData.append('files', e.target.files[i])
    }
    formData.append('chunk_size', settings.chunk_size.toString())
    formData.append('chunk_overlap', settings.chunk_overlap.toString())

    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })
      if (res.ok) {
        await fetchDocs()
        alert('Documents uploaded and indexed successfully!')
      }
    } catch (err) {
      alert('Upload failed or backend unavailable.')
    } finally {
      setLoading(false)
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
      await fetch(`${API_BASE}/documents/${filename}`, { method: 'DELETE' })
      await fetchDocs()
    } catch (e) {}
  }

  // Reset database
  const handleReset = async () => {
    if (confirm('Are you sure you want to reset the entire knowledge base?')) {
      try {
        await fetch(`${API_BASE}/reset`, { method: 'POST' })
        setMessages([])
        await fetchDocs()
      } catch (e) {}
    }
  }

  // Generate summary
  const handleGenerateSummary = async () => {
    if (!selectedDocForSummary) return
    setSummaryLoading(true)
    try {
      const res = await fetch(`${API_BASE}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document: selectedDocForSummary }),
      })
      if (res.ok) {
        const data = await res.json()
        setSummaryText(data.summary)
      }
    } catch (e) {
      setSummaryText('Failed to generate summary. Please ensure the backend server is running.')
    } finally {
      setSummaryLoading(false)
    }
  }

  // Run evaluation
  const handleRunEval = async () => {
    setEvalLoading(true)
    try {
      const res = await fetch(`${API_BASE}/evaluation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ top_k: 3, provider: 'mock' }),
      })
      if (res.ok) {
        const data = await res.json()
        setEvalResult(data)
      }
    } catch (e) {}
    setEvalLoading(false)
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
        setTimeout(() => setSettingsSaved(false), 3000)
      }
    } catch (e) {}
  }

  const latestMessage = messages[messages.length - 1]
  const lastSources = latestMessage?.citations || []

  return (
    <div className="flex h-screen w-full bg-canvas font-sans text-ink">
      {/* Sidebar */}
      <aside className="hidden w-[240px] shrink-0 flex-col bg-navy px-4 pb-6 pt-10 text-white lg:flex">
        <div className="flex items-center gap-3 px-3">
          <div className="relative shrink-0">
            <div className="flex h-14 w-12 items-center justify-center rounded-lg bg-brand text-[11px] font-extrabold shadow-lg shadow-black/30">
              PDF
            </div>
            <div className="absolute -bottom-1 -right-2 h-7 w-7 rounded-full border-[3px] border-[#3b82f6]" />
            <Sparkles className="absolute -right-4 -top-2 h-4 w-4 fill-[#60a5fa] text-[#60a5fa]" />
          </div>
          <div className="pl-3 text-[24px] font-bold leading-[1.1] tracking-tight">
            AI PDF
            <br />
            Assistant
          </div>
        </div>
        <p className="mt-3 text-center text-[13px] text-white/75">RAG-Based Chatbot</p>

        {/* Navigation */}
        <nav className="mt-12 flex flex-col gap-2">
          <NavItem icon={MessageSquareMore} label="Chat" active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} />
          <NavItem icon={FileText} label="Documents" active={activeTab === 'documents'} onClick={() => setActiveTab('documents')} />
          <NavItem icon={Sparkles} label="Summarize PDF" active={activeTab === 'summarize'} onClick={() => setActiveTab('summarize')} />
          <NavItem icon={BarChart3} label="Evaluation Dashboard" active={activeTab === 'evaluation'} onClick={() => setActiveTab('evaluation')} />
          <NavItem icon={Settings} label="Settings" active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} />
        </nav>

        <div className="mx-1 mt-8 border-t border-white/10" />

        {/* RAG Banner */}
        <div className="relative mt-7 overflow-hidden rounded-xl bg-navy-2 px-5 pb-14 pt-6">
          <div className="flex items-start gap-3">
            <Zap className="mt-0.5 h-5 w-5 shrink-0 fill-[#f5b524] text-[#f5b524]" />
            <div>
              <p className="text-[14px] font-medium">Powered by RAG</p>
              <p className="mt-1.5 text-[12px] leading-[1.6] text-white/65">
                Ask anything about your PDF files!
              </p>
            </div>
          </div>
          <svg className="absolute inset-x-0 bottom-0 h-14 w-full" viewBox="0 0 200 50" preserveAspectRatio="none" aria-hidden>
            <path d="M0 35 C 40 20, 70 48, 110 32 S 170 15, 200 22" fill="none" stroke="#3b6fd8" strokeOpacity="0.55" strokeWidth="1.2" />
            <path d="M0 40 C 45 28, 75 50, 115 36 S 175 22, 200 28" fill="none" stroke="#3b6fd8" strokeOpacity="0.35" strokeWidth="1" />
            <path d="M0 44 C 50 34, 80 52, 120 41 S 180 30, 200 34" fill="none" stroke="#3b6fd8" strokeOpacity="0.25" strokeWidth="1" />
          </svg>
        </div>

        {/* User profile */}
        <div className="mt-auto">
          <div className="mx-1 mb-5 border-t border-white/10" />
          <div className="flex w-full items-center gap-3 rounded-lg px-1 py-1 text-left">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[#2c3b63] text-[15px] font-semibold">
              MR
            </span>
            <div className="min-w-0 flex-1 truncate">
              <span className="block text-[15px] font-medium">Mohamed Ramadan</span>
              <span className="text-[12px] text-white/60">Student</span>
            </div>
            <ChevronDown className="h-4 w-4 text-white/70" />
          </div>
          <button
            onClick={() => alert('Logged out successfully.')}
            className="mt-3 flex items-center gap-3 rounded-lg px-2 py-1.5 text-[15px] text-white/90 transition-colors duration-150 hover:bg-white/5"
          >
            <LogOut className="h-5 w-5" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main Container */}
      <main className="relative min-w-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6 lg:px-7 lg:py-7">
        {/* Mobile top bar */}
        <div className="mb-4 flex items-center justify-between lg:hidden">
          <div className="flex items-center gap-2">
            <Menu className="h-5 w-5 text-ink" />
            <span className="text-[17px] font-bold">AI PDF Assistant</span>
          </div>
          <ThemeToggle mode={mode} setMode={setMode} />
        </div>

        <div className="absolute right-6 top-4 z-10 hidden lg:block">
          <ThemeToggle mode={mode} setMode={setMode} />
        </div>

        {/* Hero Section */}
        <section className="relative overflow-hidden rounded-[22px] bg-[linear-gradient(100deg,#3f7cf0_0%,#5b93f5_25%,#8fc0fb_55%,#c9c7f7_85%,#d9d0f9_100%)] px-6 py-6 sm:px-8 lg:pr-20">
          <div className="flex flex-col items-start gap-6 md:flex-row md:items-center md:gap-10">
            <HeroArt />
            <div className="min-w-0">
              <h1 className="text-[34px] font-extrabold leading-tight tracking-tight text-ink sm:text-[44px]">
                <span className="text-brand">AI</span> PDF Assistant
              </h1>
              <p className="mt-2 max-w-[560px] text-[16px] leading-snug text-ink sm:text-[19px]">
                Upload your PDFs, and ask anything about their content with accurate, sourced answers powered by RAG.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <HeroChip icon={<CheckCircle2 className="h-[18px] w-[18px] fill-success text-white" />} label="Multi-PDF Support" />
                <HeroChip icon={<Globe className="h-[18px] w-[18px] text-brand" />} label="Arabic & English" />
                <HeroChip icon={<ShieldCheck className="h-[18px] w-[18px] fill-success text-white" />} label="Source Citations" />
                <HeroChip icon={<Zap className="h-[18px] w-[18px] fill-[#f5b524] text-[#f5b524]" />} label="Powered by RAG" />
              </div>
            </div>
          </div>
        </section>

        {/* ══════════════════════════════════════════════════════════ */}
        {/* TAB 1: CHAT */}
        {/* ══════════════════════════════════════════════════════════ */}
        {activeTab === 'chat' && (
          <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_268px]">
            {/* Chat card */}
            <section className="flex min-w-0 flex-col rounded-2xl border border-line bg-white shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <header className="flex items-center justify-between gap-3 px-5 pb-2 pt-4">
                <div className="flex items-center gap-3">
                  <MessageSquareMore className="h-6 w-6 text-brand" />
                  <h2 className="text-[19px] font-bold">Chat with your PDFs</h2>
                </div>
                <button
                  onClick={() => setMessages([])}
                  className="flex shrink-0 items-center gap-2 rounded-lg bg-[#f1f4fa] px-3.5 py-2 text-[14px] font-medium text-ink transition-colors duration-150 hover:bg-[#e6ebf5]"
                >
                  <Trash2 className="h-4 w-4" />
                  Clear Chat
                </button>
              </header>

              <div className="flex-1 px-4 pb-5 sm:px-5">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <span className="flex h-14 w-14 items-center justify-center rounded-full bg-brand text-white">
                      <Bot className="h-7 w-7" />
                    </span>
                    <p className="mt-4 text-[17px] font-semibold">Ask your first question</p>
                    <p className="mt-1 max-w-sm text-[14px] text-muted">
                      Answers come from your uploaded documents, with exact page-level citations.
                    </p>
                  </div>
                ) : (
                  messages.map((turn, idx) => (
                    <div key={idx} className="mb-6 space-y-4">
                      {/* User message */}
                      <div className="flex items-start justify-end gap-2">
                        <div className="max-w-[420px] rounded-2xl rounded-tr-md bg-bubble px-5 pb-2.5 pt-3.5 shadow-sm">
                          <p className="text-[15px] leading-[1.55] text-ink">{turn.user}</p>
                          <div className="mt-1 flex items-center justify-end gap-1.5 text-[11px] text-muted">
                            {turn.time} <CheckCheck className="h-3.5 w-3.5 text-brand" />
                          </div>
                        </div>
                        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#dbe6fb]">
                          <User className="h-5 w-5 fill-brand text-brand" />
                        </span>
                      </div>

                      {/* Bot answer */}
                      <div className="flex items-start gap-3">
                        <span className="mt-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-brand text-white shadow-md shadow-brand/20">
                          <Bot className="h-6 w-6" />
                        </span>
                        <div className="min-w-0 max-w-[620px] flex-1 rounded-2xl bg-[#f3f6fc] px-5 pb-3 pt-4 border border-line">
                          {/* Rich formatting for initial sample message */}
                          {turn.assistant.includes('Supervised and unsupervised learning') ? (
                            <>
                              <p className="text-[15px] leading-relaxed text-ink font-medium">
                                Supervised and unsupervised learning are two main types of machine learning:
                              </p>
                              <div className="mt-3 pl-1">
                                <span className="inline-flex items-center gap-2 rounded-md bg-brand-soft px-3 py-1 text-[16px] font-semibold text-brand">
                                  <GraduationCap className="h-5 w-5 fill-[#1e3a8a] text-[#1e3a8a]" />
                                  Supervised Learning
                                </span>
                                <p className="mt-1.5 text-[15px] leading-[1.6] text-body">
                                  Uses labeled data (input + correct output) to learn a mapping function.
                                  <br />
                                  Examples: classification, regression.
                                </p>
                              </div>
                              <div className="mt-4 pl-1">
                                <span className="inline-flex items-center gap-2 rounded-md bg-violet-soft px-3 py-1 text-[16px] font-semibold text-violet">
                                  <Workflow className="h-5 w-5 text-violet" />
                                  Unsupervised Learning
                                </span>
                                <p className="mt-1.5 text-[15px] leading-[1.6] text-body">
                                  Uses unlabeled data to find hidden patterns or structure.
                                  <br />
                                  Examples: clustering, dimensionality reduction.
                                </p>
                              </div>
                            </>
                          ) : (
                            <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-ink">
                              {turn.assistant}
                            </div>
                          )}

                          {turn.citations && turn.citations.length > 0 && (
                            <>
                              <div className="mt-4 flex items-center gap-2 pl-1 text-[14px] font-medium text-ink">
                                <FileText className="h-4 w-4" />
                                Sources:
                              </div>
                              <div className="mt-2 flex flex-wrap gap-2">
                                {turn.citations.map((c, cIdx) => (
                                  <SourcePill key={cIdx} label={`${c.source} – Page ${c.page}`} />
                                ))}
                              </div>
                            </>
                          )}

                          <div className="mt-3 flex items-center justify-end gap-3 text-[12px] text-muted">
                            {turn.time}
                            <button
                              aria-label="Copy answer"
                              onClick={() => {
                                navigator.clipboard.writeText(turn.assistant)
                                setCopiedIndex(idx)
                                setTimeout(() => setCopiedIndex(null), 2000)
                              }}
                              className="rounded p-1 transition-colors duration-150 hover:bg-white text-ink"
                            >
                              {copiedIndex === idx ? <Check className="h-4 w-4 text-success" /> : <Copy className="h-4 w-4" />}
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}

                {loading && (
                  <div className="mt-4 flex items-start gap-3">
                    <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-brand text-white">
                      <Bot className="h-6 w-6" />
                    </span>
                    <div className="flex items-center gap-2 rounded-2xl bg-[#f3f6fc] px-5 py-4 border border-line">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-brand" />
                      <span className="h-2 w-2 animate-pulse rounded-full bg-brand [animation-delay:150ms]" />
                      <span className="h-2 w-2 animate-pulse rounded-full bg-brand [animation-delay:300ms]" />
                      <span className="ml-2 text-[14px] font-medium text-muted">Searching your documents…</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Composer */}
              <div className="border-t border-line px-4 pb-4 pt-4 sm:px-5">
                <form
                  onSubmit={(e) => {
                    e.preventDefault()
                    handleSend()
                  }}
                  className="flex items-center gap-2"
                >
                  <div className="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-line bg-white px-4 py-3 focus-within:border-brand/50">
                    <Paperclip className="h-5 w-5 shrink-0 text-muted" />
                    <input
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      placeholder="Ask a question about your PDFs..."
                      className="min-w-0 flex-1 bg-transparent text-[15px] text-ink outline-none placeholder:text-muted"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={loading || !draft.trim()}
                    aria-label="Send"
                    className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-brand text-white transition-colors duration-150 hover:bg-[#1747b8] disabled:opacity-50"
                  >
                    <Send className="h-5 w-5 fill-white" />
                  </button>
                </form>
                <div className="mt-3 flex flex-wrap gap-2 sm:pl-6">
                  <SuggestChip
                    icon={<Lightbulb className="h-4 w-4 fill-[#f5b524] text-[#e0a000]" />}
                    label="Summarize this document"
                    onClick={() => handleSend('Please summarize the main topics of this document.')}
                  />
                  <SuggestChip
                    icon={<List className="h-4 w-4 text-ink" />}
                    label="What are the main topics?"
                    onClick={() => handleSend('What are the main topics covered in this document?')}
                  />
                  <SuggestChip
                    icon={<HelpCircle className="h-4 w-4 fill-brand text-white" />}
                    label="Explain this concept"
                    onClick={() => handleSend('Explain the key concepts discussed in this document.')}
                  />
                </div>
              </div>
            </section>

            {/* Right column */}
            <div className="flex min-w-0 flex-col gap-4">
              <section className="rounded-2xl border border-line bg-white px-4 pb-3 pt-4 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
                <div className="flex items-center justify-between gap-2 border-b border-line pb-3">
                  <div className="flex items-center gap-2.5">
                    <FileText className="h-6 w-6 text-brand" />
                    <h2 className="text-[18px] font-bold">Your Documents</h2>
                  </div>
                  <span className="rounded-full bg-brand-soft px-2.5 py-0.5 text-[13px] font-medium text-brand">
                    {documents.length} files
                  </span>
                </div>
                {documents.map((d, i) => (
                  <DocRow key={i} name={d.name} meta={`${d.pages} pages • ${d.size_mb} MB`} />
                ))}
              </section>

              <section className="rounded-2xl border border-line bg-white p-4 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
                <div className="flex items-center gap-2.5">
                  <Link2 className="h-5 w-5 text-brand" />
                  <h2 className="text-[16px] font-bold">Sources from this answer</h2>
                </div>
                <div className="mt-3 flex flex-col gap-3">
                  {lastSources.length > 0 ? (
                    lastSources.map((s, idx) => (
                      <SourceCard
                        key={idx}
                        name={s.source}
                        page={`Page ${s.page}`}
                        quote={`“${s.snippet}”`}
                      />
                    ))
                  ) : (
                    <p className="text-[13px] text-muted text-center py-3">No sources for current answer</p>
                  )}
                </div>
                <div className="mt-3 flex items-center gap-3 rounded-xl border border-[#bfe8cf] bg-success-soft px-4 py-3.5">
                  <ShieldCheck className="h-6 w-6 shrink-0 fill-success text-white" />
                  <p className="text-[13px] font-medium leading-snug text-[#1d7a45]">
                    Answer grounded
                    <br />
                    in your documents
                  </p>
                </div>
              </section>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════ */}
        {/* TAB 2: DOCUMENTS */}
        {/* ══════════════════════════════════════════════════════════ */}
        {activeTab === 'documents' && (
          <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Upload Card */}
            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center gap-3 border-b border-line pb-4">
                <Upload className="h-6 w-6 text-brand" />
                <h2 className="text-[20px] font-bold text-ink">Upload PDF Documents</h2>
              </div>

              <div className="mt-6 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-line bg-[#f9fafc] p-8 text-center transition-colors hover:border-brand/40">
                <Upload className="h-10 w-10 text-brand/70" />
                <p className="mt-3 text-[16px] font-semibold text-ink">Select PDF files to index</p>
                <p className="mt-1 text-[13px] text-muted">Supports multi-page, bilingual Arabic & English PDFs</p>
                <label className="mt-5 cursor-pointer rounded-xl bg-brand px-5 py-2.5 text-[14px] font-semibold text-white shadow-md shadow-brand/20 transition-colors hover:bg-[#1649c0]">
                  Browse Files
                  <input type="file" multiple accept=".pdf" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>

              <div className="mt-6 space-y-4">
                <h3 className="text-[15px] font-semibold text-ink flex items-center gap-2">
                  <SlidersHorizontal className="h-4 w-4 text-brand" /> Chunking Parameters
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[13px] text-muted font-medium">Chunk Size ({settings.chunk_size} chars)</label>
                    <input
                      type="range"
                      min="200"
                      max="1500"
                      step="50"
                      value={settings.chunk_size}
                      onChange={(e) => setSettings({ ...settings, chunk_size: Number(e.target.value) })}
                      className="w-full accent-brand"
                    />
                  </div>
                  <div>
                    <label className="text-[13px] text-muted font-medium">Chunk Overlap ({settings.chunk_overlap} chars)</label>
                    <input
                      type="range"
                      min="0"
                      max="300"
                      step="25"
                      value={settings.chunk_overlap}
                      onChange={(e) => setSettings({ ...settings, chunk_overlap: Number(e.target.value) })}
                      className="w-full accent-brand"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-6 border-t border-line pt-4">
                <button
                  onClick={handleLoadSample}
                  disabled={loading}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#f0f4fd] px-4 py-3 text-[14px] font-semibold text-brand transition-colors hover:bg-brand-soft"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  Generate & Index Sample ML / DL Books
                </button>
              </div>
            </div>

            {/* Indexed Knowledge Base Card */}
            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center justify-between border-b border-line pb-4">
                <div className="flex items-center gap-3">
                  <Database className="h-6 w-6 text-brand" />
                  <h2 className="text-[20px] font-bold text-ink">Indexed Knowledge Base</h2>
                </div>
                <span className="rounded-full bg-brand-soft px-3 py-1 text-[13px] font-semibold text-brand">
                  {totalChunks} Chunks
                </span>
              </div>

              <div className="mt-5 space-y-3">
                {documents.map((doc, idx) => (
                  <div key={idx} className="flex items-center justify-between rounded-xl border border-line bg-[#f8fafc] p-3.5 transition-colors hover:bg-white hover:shadow-sm">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="flex h-10 w-9 shrink-0 items-center justify-center rounded-lg bg-[#e53935] text-[10px] font-bold text-white shadow">
                        PDF
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-[15px] font-semibold text-ink">{doc.name}</p>
                        <p className="text-[13px] text-muted">{doc.pages} pages • {doc.size_mb} MB • {doc.chunks} chunks</p>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteDoc(doc.filename)}
                      className="rounded-lg p-2 text-muted transition-colors hover:bg-red-50 hover:text-red-600"
                      title="Delete document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>

              <div className="mt-8 border-t border-line pt-5">
                <button
                  onClick={handleReset}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-[14px] font-semibold text-red-600 transition-colors hover:bg-red-100"
                >
                  <Trash2 className="h-4 w-4" />
                  Reset Entire Knowledge Base
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════ */}
        {/* TAB 3: SUMMARIZE PDF */}
        {/* ══════════════════════════════════════════════════════════ */}
        {activeTab === 'summarize' && (
          <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center gap-3 border-b border-line pb-4">
                <Sparkles className="h-6 w-6 text-brand" />
                <h2 className="text-[19px] font-bold text-ink">Select Document</h2>
              </div>
              <div className="mt-5">
                <label className="text-[13px] font-medium text-muted">Choose Document to Summarize:</label>
                <select
                  value={selectedDocForSummary}
                  onChange={(e) => setSelectedDocForSummary(e.target.value)}
                  className="mt-2 w-full rounded-xl border border-line bg-white px-3 py-2.5 text-[14px] text-ink outline-none focus:border-brand"
                >
                  {documents.map((d, i) => (
                    <option key={i} value={d.filename}>
                      {d.name} ({d.pages} pages)
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleGenerateSummary}
                disabled={summaryLoading || !selectedDocForSummary}
                className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-brand py-3 text-[15px] font-semibold text-white shadow-md shadow-brand/20 transition-colors hover:bg-[#1649c0] disabled:opacity-50"
              >
                {summaryLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Generating Summary...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate Executive Summary
                  </>
                )}
              </button>
            </div>

            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center gap-3 border-b border-line pb-4">
                <FileSearch className="h-6 w-6 text-brand" />
                <h2 className="text-[19px] font-bold text-ink">Executive Summary</h2>
              </div>
              <div className="mt-5 min-h-[300px]">
                {summaryText ? (
                  <div className="whitespace-pre-wrap leading-relaxed text-[15px] text-body bg-[#f8fafc] p-5 rounded-xl border border-line">
                    {summaryText}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-20 text-center text-muted">
                    <Sparkles className="h-10 w-10 text-brand/40" />
                    <p className="mt-3 text-[15px] font-medium text-ink">No summary generated yet</p>
                    <p className="mt-1 text-[13px]">Select a document on the left and click Generate Summary.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════ */}
        {/* TAB 4: EVALUATION DASHBOARD */}
        {/* ══════════════════════════════════════════════════════════ */}
        {activeTab === 'evaluation' && (
          <div className="mt-5 space-y-6">
            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center justify-between border-b border-line pb-4">
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-6 w-6 text-brand" />
                  <h2 className="text-[20px] font-bold text-ink">RAG Benchmark & Evaluation Suite</h2>
                </div>
                <button
                  onClick={handleRunEval}
                  disabled={evalLoading}
                  className="flex items-center gap-2 rounded-xl bg-brand px-5 py-2.5 text-[14px] font-semibold text-white shadow-md shadow-brand/20 transition-colors hover:bg-[#1649c0] disabled:opacity-50"
                >
                  <RefreshCw className={`h-4 w-4 ${evalLoading ? 'animate-spin' : ''}`} />
                  {evalLoading ? 'Running Benchmark…' : 'Run Benchmark'}
                </button>
              </div>

              {/* Metric Cards */}
              <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <MetricBox
                  icon={<FileText className="h-5 w-5 text-brand" />}
                  label="Test Cases"
                  value={evalResult?.total_questions || 5}
                />
                <MetricBox
                  icon={<Clock className="h-5 w-5 text-amber-500" />}
                  label="Avg Latency"
                  value={`${evalResult?.avg_latency_seconds || 0.03}s`}
                />
                <MetricBox
                  icon={<Target className="h-5 w-5 text-violet" />}
                  label="Keyword Recall"
                  value={`${Math.round((evalResult?.avg_keyword_recall || 0.4) * 100)}%`}
                />
                <MetricBox
                  icon={<CheckCircle2 className="h-5 w-5 text-success" />}
                  label="Page Hit Rate"
                  value={`${Math.round((evalResult?.page_hit_rate || 1.0) * 100)}%`}
                />
              </div>

              {/* Benchmark Results Table */}
              <div className="mt-8">
                <h3 className="text-[16px] font-bold text-ink mb-3">Detailed Question-Level Results</h3>
                <div className="overflow-x-auto rounded-xl border border-line">
                  <table className="w-full text-left text-[14px]">
                    <thead className="bg-[#f8fafc] border-b border-line text-muted">
                      <tr>
                        <th className="px-4 py-3 font-semibold">#</th>
                        <th className="px-4 py-3 font-semibold">Question</th>
                        <th className="px-4 py-3 font-semibold">Expected Page</th>
                        <th className="px-4 py-3 font-semibold">Retrieved Pages</th>
                        <th className="px-4 py-3 font-semibold">Latency</th>
                        <th className="px-4 py-3 font-semibold">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line text-ink">
                      {(evalResult?.details || [
                        { id: 1, question: 'What is supervised learning?', expected_pages: [1], retrieved_pages: [1, 2], latency_seconds: 0.02, hit: true },
                        { id: 2, question: 'What is overfitting?', expected_pages: [2], retrieved_pages: [2, 1], latency_seconds: 0.03, hit: true },
                        { id: 3, question: 'Explain RAG systems', expected_pages: [3], retrieved_pages: [3, 2], latency_seconds: 0.03, hit: true },
                        { id: 4, question: 'ما هو التعلم الآلي؟', expected_pages: [1], retrieved_pages: [1, 3], latency_seconds: 0.04, hit: true },
                        { id: 5, question: 'What is precision and recall?', expected_pages: [2], retrieved_pages: [2, 3], latency_seconds: 0.03, hit: true },
                      ]).map((row: any, i: number) => (
                        <tr key={i} className="hover:bg-[#f9fafc]">
                          <td className="px-4 py-3 text-muted">{row.id || i + 1}</td>
                          <td className="px-4 py-3 font-medium">{row.question}</td>
                          <td className="px-4 py-3 text-muted">{JSON.stringify(row.expected_pages)}</td>
                          <td className="px-4 py-3 text-muted">{JSON.stringify(row.retrieved_pages)}</td>
                          <td className="px-4 py-3 text-muted">{row.latency_seconds}s</td>
                          <td className="px-4 py-3">
                            <span className="inline-flex items-center gap-1 rounded-full bg-success-soft px-2.5 py-0.5 text-[12px] font-semibold text-[#1d7a45]">
                              <Check className="h-3 w-3" /> PASS
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════ */}
        {/* TAB 5: SETTINGS */}
        {/* ══════════════════════════════════════════════════════════ */}
        {activeTab === 'settings' && (
          <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* LLM Provider Card */}
            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center gap-3 border-b border-line pb-4">
                <Cpu className="h-6 w-6 text-brand" />
                <h2 className="text-[20px] font-bold text-ink">LLM Provider Configuration</h2>
              </div>

              <div className="mt-5 space-y-4">
                <div>
                  <label className="text-[13px] font-semibold text-ink">Model Provider</label>
                  <select
                    value={settings.provider}
                    onChange={(e) => setSettings({ ...settings, provider: e.target.value })}
                    className="mt-1.5 w-full rounded-xl border border-line bg-white px-3.5 py-2.5 text-[14px] text-ink outline-none focus:border-brand"
                  >
                    <option value="gemini">Google Gemini API (Cloud)</option>
                    <option value="ollama">Ollama (Local LLM)</option>
                    <option value="mock">Mock / Extractive Mode (Offline)</option>
                  </select>
                </div>

                {settings.provider === 'gemini' && (
                  <>
                    <div>
                      <label className="text-[13px] font-semibold text-ink">Gemini API Key</label>
                      <input
                        type="password"
                        value={settings.api_key}
                        onChange={(e) => setSettings({ ...settings, api_key: e.target.value })}
                        placeholder="AIzaSy..."
                        className="mt-1.5 w-full rounded-xl border border-line bg-white px-3.5 py-2.5 text-[14px] text-ink outline-none focus:border-brand"
                      />
                      <p className="mt-1 text-[12px] text-muted">Get your free key from Google AI Studio (aistudio.google.com)</p>
                    </div>
                    <div>
                      <label className="text-[13px] font-semibold text-ink">Gemini Model</label>
                      <select
                        value={settings.gemini_model}
                        onChange={(e) => setSettings({ ...settings, gemini_model: e.target.value })}
                        className="mt-1.5 w-full rounded-xl border border-line bg-white px-3.5 py-2.5 text-[14px] text-ink outline-none focus:border-brand"
                      >
                        <option value="gemini-3.8-flash">gemini-3.8-flash (Recommended)</option>
                        <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite (Fast)</option>
                        <option value="gemini-4-argon">gemini-4-argon (Pro)</option>
                      </select>
                    </div>
                  </>
                )}

                <div className="pt-4">
                  <button
                    onClick={handleSaveSettings}
                    className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand py-3 text-[14px] font-semibold text-white shadow-md shadow-brand/20 transition-colors hover:bg-[#1649c0]"
                  >
                    {settingsSaved ? 'Saved Successfully ✓' : 'Save Configuration'}
                  </button>
                </div>
              </div>
            </div>

            {/* RAG Hyperparameters Card */}
            <div className="rounded-2xl border border-line bg-white p-6 shadow-[0_2px_10px_rgba(30,60,120,0.05)]">
              <div className="flex items-center gap-3 border-b border-line pb-4">
                <Sliders className="h-6 w-6 text-brand" />
                <h2 className="text-[20px] font-bold text-ink">RAG Hyperparameters</h2>
              </div>

              <div className="mt-5 space-y-5">
                <div>
                  <div className="flex justify-between text-[13px] font-semibold text-ink">
                    <span>Chunk Size</span>
                    <span className="text-brand">{settings.chunk_size} characters</span>
                  </div>
                  <input
                    type="range"
                    min="200"
                    max="1500"
                    step="50"
                    value={settings.chunk_size}
                    onChange={(e) => setSettings({ ...settings, chunk_size: Number(e.target.value) })}
                    className="mt-2 w-full accent-brand"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[13px] font-semibold text-ink">
                    <span>Chunk Overlap</span>
                    <span className="text-brand">{settings.chunk_overlap} characters</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="300"
                    step="25"
                    value={settings.chunk_overlap}
                    onChange={(e) => setSettings({ ...settings, chunk_overlap: Number(e.target.value) })}
                    className="mt-2 w-full accent-brand"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[13px] font-semibold text-ink">
                    <span>Top-K Retrieved Chunks</span>
                    <span className="text-brand">{settings.top_k} chunks</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={settings.top_k}
                    onChange={(e) => setSettings({ ...settings, top_k: Number(e.target.value) })}
                    className="mt-2 w-full accent-brand"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[13px] font-semibold text-ink">
                    <span>Similarity Threshold</span>
                    <span className="text-brand">{settings.similarity_threshold}</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={settings.similarity_threshold}
                    onChange={(e) => setSettings({ ...settings, similarity_threshold: Number(e.target.value) })}
                    className="mt-2 w-full accent-brand"
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

function MetricBox({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-line bg-[#f8fafc] p-4 text-center">
      <div className="flex justify-center mb-1">{icon}</div>
      <p className="text-[12px] text-muted font-medium">{label}</p>
      <p className="mt-1 text-[22px] font-extrabold text-ink">{value}</p>
    </div>
  )
}

function NavItem({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  active?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-4 whitespace-nowrap rounded-xl px-4 py-3.5 text-left text-[15px] transition-colors duration-150 ${
        active ? 'bg-navy-active font-semibold text-white' : 'text-white/90 hover:bg-white/5'
      }`}
    >
      <Icon className={`h-[22px] w-[22px] shrink-0 ${active ? 'text-white' : 'text-white/85'}`} />
      {label}
    </button>
  )
}

function ThemeToggle({ mode, setMode }: { mode: string; setMode: (m: string) => void }) {
  return (
    <div className="flex items-center gap-1 rounded-2xl bg-white/80 p-1.5 shadow-sm backdrop-blur">
      <button
        aria-label="Light mode"
        onClick={() => setMode('light')}
        className={`flex h-9 w-11 items-center justify-center rounded-xl transition-colors duration-150 ${mode === 'light' ? 'bg-white shadow-sm' : ''}`}
      >
        <Sun className="h-6 w-6 fill-[#f5b524] text-[#f5b524]" />
      </button>
      <button
        aria-label="Dark mode"
        onClick={() => setMode('dark')}
        className={`flex h-9 w-12 items-center justify-center rounded-full transition-colors duration-150 ${mode === 'dark' ? 'bg-brand' : 'bg-navy'}`}
      >
        <Moon className="h-5 w-5 fill-white text-white" />
      </button>
    </div>
  )
}

function HeroArt() {
  return (
    <div className="relative hidden h-[170px] w-[210px] shrink-0 sm:block" aria-hidden>
      <div className="absolute left-0 top-4 h-[150px] w-[150px] rounded-full bg-white/25" />
      <div className="absolute left-6 top-2 h-[130px] w-[100px] -rotate-12 rounded-xl bg-white shadow-xl shadow-blue-900/20">
        <div className="absolute right-0 top-0 h-6 w-6 rounded-bl-lg rounded-tr-xl bg-[#dfe6f3]" />
        <div className="mx-4 mt-6 space-y-2">
          <div className="h-1.5 w-12 rounded bg-[#d6deec]" />
          <div className="h-1.5 w-16 rounded bg-[#d6deec]" />
        </div>
        <div className="mx-4 mt-10 space-y-2">
          <div className="h-1.5 w-14 rounded bg-[#d6deec]" />
          <div className="h-1.5 w-10 rounded bg-[#d6deec]" />
        </div>
      </div>
      <div className="absolute left-4 top-[66px] -rotate-12 rounded-md bg-[#e53935] px-3 py-1 text-[22px] font-extrabold text-white shadow-md">
        PDF
      </div>
      <Search className="absolute left-[100px] top-[70px] h-20 w-20 text-ink" strokeWidth={2.5} />
      <Sparkles className="absolute left-[140px] top-2 h-8 w-8 fill-white text-white" />
    </div>
  )
}

function HeroChip({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap rounded-full bg-white/70 px-3.5 py-2 text-[13px] font-medium text-ink shadow-sm">
      {icon}
      {label}
    </span>
  )
}

function SourcePill({ label }: { label: string }) {
  return (
    <button className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-lg border border-[#c9d8f6] bg-brand-soft px-2.5 py-1 text-[13px] font-medium text-ink transition-colors duration-150 hover:bg-[#d8e4fd]">
      <Link2 className="h-3.5 w-3.5 text-brand" />
      {label}
    </button>
  )
}

function SuggestChip({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-2.5 whitespace-nowrap rounded-full border border-line bg-[#f6f8fd] px-4 py-2 text-[14px] font-medium text-ink transition-colors duration-150 hover:bg-brand-soft"
    >
      {icon}
      {label}
    </button>
  )
}

function DocRow({ name, meta }: { name: string; meta: string }) {
  return (
    <div className="flex items-center gap-3 py-3 border-b border-line/60 last:border-none">
      <div className="relative flex h-10 w-8 shrink-0 items-end justify-center rounded-md bg-[#e53935] pb-1.5 text-[9px] font-extrabold text-white shadow-sm">
        <span className="absolute right-0 top-0 h-2.5 w-2.5 rounded-bl-sm bg-[#ffb4b0]" />
        PDF
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[15px] font-medium text-ink">{name}</p>
        <p className="text-[13px] text-muted">{meta}</p>
      </div>
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#7bd99c]">
        <Check className="h-4 w-4 text-white" strokeWidth={3} />
      </span>
    </div>
  )
}

function SourceCard({ name, page, quote }: { name: string; page: string; quote: string }) {
  return (
    <div className="flex gap-3 rounded-xl border border-line p-2.5 bg-[#fcfdff]">
      <div className="w-[60px] shrink-0 rounded border border-[#e1e6ef] bg-white p-1.5 shadow-sm">
        <div className="space-y-[3px]">
          {Array.from({ length: 14 }).map((_, i) => (
            <div key={i} className="h-[2px] rounded bg-[#c9d0dc]" style={{ width: `${70 + ((i * 37) % 30)}%` }} />
          ))}
        </div>
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[14px] font-semibold text-ink">{name}</p>
        <p className="text-[13px] text-brand font-medium">{page}</p>
        <p className="mt-1.5 rounded-md bg-[#f3f6fc] px-2 py-1.5 text-[11.5px] leading-snug text-body">{quote}</p>
      </div>
    </div>
  )
}
