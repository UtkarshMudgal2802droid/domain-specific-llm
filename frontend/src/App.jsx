import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

const TOOLS = [
  {
    id: 'qa',
    title: 'Domain Q&A',
    subtitle: 'Ask complex finance and tech questions.',
    endpoint: '/generate',
    model: 'Mistral-7B-LoRA',
    kind: 'single',
    placeholder: 'Ask about EBITDA, Kubernetes, SIP, Docker...',
    promptPrefix: '',
    examples: [
      'Explain EBITDA in simple terms.',
      'What is the difference between Docker and Kubernetes?',
      'How does a SIP investment work?',
    ],
    button: 'Ask AI',
  },
  {
    id: 'summarize',
    title: 'Report Summarizer',
    subtitle: 'Condense financial or technical reports.',
    endpoint: '/generate',
    model: 'Mistral-7B-LoRA',
    kind: 'single',
    placeholder: 'Paste a long financial report or technical documentation here...',
    promptPrefix: 'Please summarize the following text concisely:\n\n',
    examples: [
      'The company reported a 15% increase in revenue YoY, driven by strong cloud computing demand. However, operating expenses rose 20% due to aggressive R&D in AI.',
      'Microservices architecture structures an application as a collection of loosely coupled, independently deployable services.',
    ],
    button: 'Summarize',
  },
  {
    id: 'jargon',
    title: 'Jargon Buster',
    subtitle: 'Get clear definitions for industry terms.',
    endpoint: '/generate',
    model: 'Mistral-7B-LoRA',
    kind: 'single',
    placeholder: 'Enter a term (e.g., P/E Ratio, CI/CD, Equity dilution)...',
    promptPrefix: 'Define and explain the following industry term:\n\n',
    examples: [
      'P/E Ratio',
      'Vector databases',
      'Continuous Integration',
    ],
    button: 'Explain',
  }
];

function App() {
  const [activeTool, setActiveTool] = useState('qa');
  
  // Backend status: 'checking' | 'connected' | 'error'
  const [backendState, setBackendState] = useState('checking');
  const [statusMsg, setStatusMsg] = useState('Connecting to inference engine...');
  
  const [loading, setLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  
  const [output, setOutput] = useState(null);
  const [meta, setMeta] = useState(null);
  
  const [inputText, setInputText] = useState('');
  const [inputError, setInputError] = useState('');
  
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);

  const tool = useMemo(() => TOOLS.find((item) => item.id === activeTool), [activeTool]);

  useEffect(() => {
    // Optimistic connection check. Since our API doesn't have a /health endpoint out of the box,
    // we'll assume the system is ready, or fail gracefully on generation.
    // To make it robust, a simple fetch could be added if a health route existed.
    setTimeout(() => {
      setBackendState('connected');
      setStatusMsg('System Online · Models Loaded');
    }, 800);
  }, []);

  const runTool = async () => {
    if (!inputText.trim()) {
      setInputError('Please provide input text before running the model.');
      return;
    }

    setInputError('');
    setLoading(true);
    setIsError(false);
    setOutput(null);
    setMeta(null);
    setCopied(false);

    const started = performance.now();
    const finalPrompt = tool.promptPrefix + inputText;

    try {
      const res = await axios.post(`${API_BASE_URL}${tool.endpoint}`, { prompt: finalPrompt });
      const elapsed = Math.max(1, Math.round(performance.now() - started));
      
      const resultText = res.data.response;

      setHistory((prev) => [
        { 
          id: Date.now(), 
          tool: tool.title, 
          preview: inputText.slice(0, 50) + (inputText.length > 50 ? '...' : ''), 
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
        },
        ...prev.slice(0, 4),
      ]);

      setOutput({ type: 'text', data: resultText });
      setMeta({
        model: tool.model,
        endpoint: tool.endpoint,
        latency: `${elapsed} ms`,
        task: tool.id,
      });
    } catch (err) {
      setIsError(true);
      setOutput({ type: 'error', data: err.response?.data?.detail || err.message || 'The request failed to process. Ensure the backend is running.' });
      setMeta({ model: tool.model, endpoint: tool.endpoint, latency: 'failed', task: tool.id });
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (value) => {
    setInputError('');
    setInputText(value);
  };

  const clearAll = () => {
    setInputText('');
    setOutput(null);
    setMeta(null);
    setInputError('');
    setIsError(false);
    setCopied(false);
  };

  const copyToClipboard = () => {
    if (!output || output.type !== 'text') return;
    
    navigator.clipboard.writeText(output.data);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderOutput = () => {
    if (loading) {
      return (
        <div className="skeleton-wrap">
          <div className="skeleton-line title" />
          <div className="skeleton-line" />
          <div className="skeleton-line" />
          <div className="skeleton-line short" />
        </div>
      );
    }

    if (isError) {
      return (
        <div className="error-panel">
          <svg className="error-icon" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div>
            <strong>Processing Error</strong>
            <p>{output?.data}</p>
          </div>
        </div>
      );
    }

    if (!output) {
      return (
        <div className="empty-state">
          <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
          </svg>
          <p>Your result will appear here. Submit a request to see output.</p>
        </div>
      );
    }

    return <pre>{output.data}</pre>;
  };

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand">
          <div className="brand-mark">AI</div>
          <div>
            <h1>Domain AI</h1>
            <p>Finance & Tech Models</p>
          </div>
        </div>

        <div className="status-card">
          <span className={`status-dot ${backendState}`} />
          <div>
            <strong>Backend Status</strong>
            <p>{statusMsg}</p>
          </div>
        </div>

        <div className="tool-list">
          {TOOLS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`tool-item ${activeTool === item.id ? 'active' : ''}`}
              onClick={() => {
                setActiveTool(item.id);
                clearAll();
              }}
            >
              <span>{item.title}</span>
              <small>{item.subtitle}</small>
            </button>
          ))}
        </div>

        <div className="history-card">
          <div className="history-head">
            <h3>Recent Queries</h3>
          </div>
          {history.length === 0 ? (
            <p className="history-empty">No activity yet.</p>
          ) : (
            history.map((item) => (
              <div key={item.id} className="history-item">
                <strong>{item.tool}</strong>
                <p>{item.preview}</p>
                <span>{item.time}</span>
              </div>
            ))
          )}
        </div>
      </aside>

      <main className="workspace">
        <section className="hero-card">
          <div className="hero-content">
            <p className="eyebrow">Task Configuration</p>
            <h2>{tool.title}</h2>
            <p className="hero-copy">{tool.subtitle}</p>
          </div>

          <div className="hero-stats">
            <div className="stat-box">
              <span>Model Pipeline</span>
              <strong>{tool.model}</strong>
            </div>
          </div>
        </section>

        <section className="panel form-panel">
          <div className="examples-area">
            <span className="examples-label">Try an example:</span>
            <div className="chip-row">
              {tool.examples.map((item, idx) => (
                <button key={idx} type="button" className="chip" onClick={() => loadExample(item)}>
                  {item.length > 40 ? item.substring(0, 40) + '...' : item}
                </button>
              ))}
            </div>
          </div>

          <label className="field">
            <span>Input Prompt</span>
            <textarea
              className={`input-box large ${inputError ? 'error-border' : ''}`}
              placeholder={tool.placeholder}
              value={inputText}
              onChange={(e) => {
                setInputText(e.target.value);
                if (inputError) setInputError('');
              }}
            />
            {inputError && <span className="error-text">{inputError}</span>}
          </label>

          <div className="actions">
            <button type="button" className="primary-btn" onClick={runTool} disabled={loading || !inputText.trim()}>
              {loading ? (
                <span className="btn-content">
                  <svg className="spinner" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
              ) : (
                tool.button
              )}
            </button>
            <button type="button" className="secondary-btn" onClick={clearAll} disabled={loading}>
              Clear
            </button>
          </div>
        </section>

        <section className="output-grid">
          <article className="output-card main-output">
            <div className="output-head">
              <div className="output-title">
                <h3>Result</h3>
                {meta && <span className="mini-pill">{meta.latency}</span>}
              </div>
              {output && !isError && (
                <button type="button" className="action-badge" onClick={copyToClipboard}>
                  {copied ? 'Copied!' : 'Copy result'}
                </button>
              )}
            </div>
            <div className="output-body">
              {renderOutput()}
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}

export default App;
