import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  Camera,
  ChevronDown,
  Loader2,
  MessageSquareText,
  Send,
  Sparkles,
  UserRound,
} from "lucide-react";
import "./styles.css";

const STARTER_PROMPTS = [
  "Where are the workers in d23 and tell me the situation",
  "What were the busiest areas in d23?",
  "Were workers gathered near one place?",
  "Which areas look underused?",
];

function makeSessionId() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return `session_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text:
        "Ask me about worker movement, busy areas, gathering points, or press-side activity. I will use the CCTV tools and keep the answer in plain floor-manager language.",
    },
  ]);
  const [input, setInput] = useState("");
  const [camera, setCamera] = useState("d23");
  const [provider, setProvider] = useState("openai");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState(makeSessionId);
  const scrollRef = useRef(null);

  const canSend = input.trim().length > 0 && !isStreaming;

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isStreaming]);

  const statusText = useMemo(() => {
    if (isStreaming) return "Agent is checking tools and drafting the brief";
    return `Ready on camera ${camera || "not set"}`;
  }, [camera, isStreaming]);

  async function sendMessage(text = input) {
    const clean = text.trim();
    if (!clean || isStreaming) return;

    setInput("");
    setIsStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: clean },
      { role: "assistant", text: "", streaming: true },
    ]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: clean,
          camera,
          provider,
          session_id: sessionId,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);

          if (event.type === "session" && event.session_id) {
            setSessionId(event.session_id);
          }

          if (event.type === "chunk") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              next[next.length - 1] = {
                ...last,
                text: `${last.text}${event.text}`,
                streaming: true,
              };
              return next;
            });
          }

          if (event.type === "image") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              next[next.length - 1] = {
                ...last,
                images: [...(last.images || []), { url: event.url, name: event.name }],
                streaming: true,
              };
              return next;
            });
          }

          if (event.type === "error") {
            throw new Error(event.error || "Agent error");
          }
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          text: `I could not complete that request: ${error.message}`,
        };
        return next;
      });
    } finally {
      setIsStreaming(false);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, streaming: false };
        }
        return next;
      });
    }
  }

  function resetChat() {
    setSessionId(makeSessionId());
    setMessages([
      {
        role: "assistant",
        text:
          "New chat started. Ask me what is happening on the floor and I will summarize it from the CCTV data.",
      },
    ]);
  }

  return (
    <main className="app-shell">
      <section className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            <Sparkles size={22} />
          </div>
          <div>
            <h1>CCTV Intelligence</h1>
            <p>Factory floor agent</p>
          </div>
        </div>

        <div className="control-panel">
          <label>
            <span>
              <Camera size={15} />
              Camera
            </span>
            <input value={camera} onChange={(event) => setCamera(event.target.value)} />
          </label>

          <label>
            <span>
              <Bot size={15} />
              Provider
            </span>
            <div className="select-wrap">
              <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini</option>
              </select>
              <ChevronDown size={16} />
            </div>
          </label>
        </div>

        <div className="prompt-stack">
          <p>Quick asks</p>
          {STARTER_PROMPTS.map((prompt) => (
            <button key={prompt} type="button" onClick={() => sendMessage(prompt)} disabled={isStreaming}>
              {prompt}
            </button>
          ))}
        </div>

        <button className="reset-button" type="button" onClick={resetChat} disabled={isStreaming}>
          New chat
        </button>
      </section>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Live agent</p>
            <h2>Floor Situation Chat</h2>
          </div>
          <div className="status-pill">
            {isStreaming ? <Loader2 size={15} className="spin" /> : <MessageSquareText size={15} />}
            {statusText}
          </div>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <div className="avatar">
                {message.role === "assistant" ? <Bot size={17} /> : <UserRound size={17} />}
              </div>
              <div className="bubble">
                <div className="message-label">{message.role === "assistant" ? "Agent" : "You"}</div>
                <p>
                  {message.text}
                  {message.streaming ? <span className="cursor" /> : null}
                </p>
                {message.images?.length ? (
                  <div className="image-strip">
                    {message.images.map((image) => (
                      <a href={image.url} target="_blank" rel="noreferrer" key={image.url}>
                        <img src={image.url} alt={image.name || "CCTV reference"} />
                        <span>{image.name || "CCTV reference"}</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            </article>
          ))}
          <div ref={scrollRef} />
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            sendMessage();
          }}
        >
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Ask about worker movement, busy spots, gathering points, or workstation activity..."
            rows={1}
          />
          <button type="submit" disabled={!canSend} title="Send message">
            {isStreaming ? <Loader2 size={19} className="spin" /> : <Send size={19} />}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
