import { useState, useRef, useEffect, useCallback } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  MessageCircle, Send, Bot, User, Sparkles, Mic, MicOff, Loader2, Globe,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { chatAPI } from "@/services/api";

// ─── Types ────────────────────────────────────────────────────────────────────
interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

interface Language {
  code: string;
  label: string;
  native: string;
}

// ─── Speech Recognition Types (fixes all TS errors) ──────────────────────────
interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}

interface ISpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: ((this: ISpeechRecognition, ev: Event) => void) | null;
  onend: ((this: ISpeechRecognition, ev: Event) => void) | null;
  onresult: ((this: ISpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
  onerror: ((this: ISpeechRecognition, ev: SpeechRecognitionErrorEvent) => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition: new () => ISpeechRecognition;
    webkitSpeechRecognition: new () => ISpeechRecognition;
  }
}

// ─── Constants ────────────────────────────────────────────────────────────────
const LANGUAGES: Language[] = [
  { code: "en-IN", label: "English",  native: "English"  },
  { code: "hi-IN", label: "Hindi",    native: "हिन्दी"    },
  { code: "bn-IN", label: "Bengali",  native: "বাংলা"    },
  { code: "te-IN", label: "Telugu",   native: "తెలుగు"   },
  { code: "mr-IN", label: "Marathi",  native: "मराठी"    },
  { code: "ta-IN", label: "Tamil",    native: "தமிழ்"    },
  { code: "gu-IN", label: "Gujarati", native: "ગુજરાતી"  },
  { code: "kn-IN", label: "Kannada",  native: "ಕನ್ನಡ"    },
  { code: "pa-IN", label: "Punjabi",  native: "ਪੰਜਾਬੀ"   },
  { code: "or-IN", label: "Odia",     native: "ଓଡ଼ିଆ"    },
];

const quickQuestions = [
  "What crops should I plant this season?",
  "How to treat leaf blight in tomatoes?",
  "Best fertilizer for wheat crops",
  "When to irrigate rice fields?",
];

const DEFAULT_GREETING =
  "I'm KrishiSahyog AI, your agriculture assistant! I can help with crop recommendations, disease treatment, soil management, and more. What would you like to know?";

// ─── Component ────────────────────────────────────────────────────────────────
const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", type: "bot", content: DEFAULT_GREETING, timestamp: new Date() },
  ]);
  const [input, setInput]             = useState("");
  const [isTyping, setIsTyping]       = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [selectedLang, setSelectedLang] = useState<Language>(LANGUAGES[0]);
  const [showLangMenu, setShowLangMenu] = useState(false);
  const [voiceError, setVoiceError]   = useState<string | null>(null);
  const [interimText, setInterimText] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<ISpeechRecognition | null>(null);
  const langMenuRef    = useRef<HTMLDivElement>(null);

  // ─── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ─── Close language menu on outside click ──────────────────────────────────
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (langMenuRef.current && !langMenuRef.current.contains(e.target as Node)) {
        setShowLangMenu(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // ─── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => { recognitionRef.current?.abort(); };
  }, []);

  // ─── Voice Recognition ─────────────────────────────────────────────────────
  const startListening = useCallback(() => {
    setVoiceError(null);

    const SpeechRecognitionAPI =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      setVoiceError("Voice input is not supported in this browser. Try Chrome or Edge.");
      return;
    }

    // Toggle off if already listening
    if (isListening) {
      recognitionRef.current?.stop();
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognitionRef.current = recognition;

    recognition.lang            = selectedLang.code;
    recognition.continuous      = true;   // ✅ stays on through natural pauses
    recognition.interimResults  = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setInterimText("");
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      let final   = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += transcript;
        } else {
          interim += transcript;
        }
      }

      if (interim) setInterimText(interim);
      if (final) {
        setInput((prev) => (prev ? prev + " " + final : final).trim());
        setInterimText("");
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setIsListening(false);
      setInterimText("");

      if (event.error === "not-allowed") {
        setVoiceError("Microphone permission denied. Please allow mic access in your browser.");
      } else if (event.error === "no-speech") {
        setVoiceError("No speech detected. Please try again.");
      } else if (event.error === "network") {
        setVoiceError("Network error during voice recognition.");
      } else {
        setVoiceError(`Voice error: ${event.error}`);
      }

      setTimeout(() => setVoiceError(null), 4000);
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimText("");
    };

    try {
      recognition.start();
    } catch {
      setVoiceError("Could not start microphone. Please try again.");
      setIsListening(false);
    }
  }, [isListening, selectedLang]);

  // ─── Send Message ──────────────────────────────────────────────────────────
  const sendMessage = async () => {
    if (!input.trim()) return;

    if (isListening) recognitionRef.current?.stop();

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const query = input;
    setInput("");
    setInterimText("");
    setIsTyping(true);

    try {
      const history = messages
        .filter((m) => m.type === "user" || m.type === "bot")
        .map((m) => ({
          role: m.type === "user" ? "user" : "assistant",
          content: m.content,
        }));

      const { data } = await chatAPI.send(query, history);

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: data.response || "Sorry, I couldn't generate a response. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        "Chat failed.";
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: `Error: ${msg}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleQuickQuestion = (question: string) => {
    setInput(question);
  };

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <DashboardLayout>
      <div className="h-[calc(100vh-8rem)] flex flex-col">

        {/* Page Header */}
        <div className="mb-4">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <MessageCircle className="h-8 w-8 text-primary" />
            Smart Assistant
          </h1>
          <p className="text-muted-foreground mt-1">
            Get personalized farming advice in your language
          </p>
        </div>

        {/* Chat Card */}
        <Card className="flex-1 flex flex-col overflow-hidden">

          {/* Card Header — bot info + language picker */}
          <CardHeader className="border-b py-3">
            <div className="flex items-center justify-between">

              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle className="text-base">Krishi AI</CardTitle>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
                    Online • Ready to help
                  </p>
                </div>
              </div>

              {/* Language Selector */}
              <div className="relative" ref={langMenuRef}>
                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2 text-xs"
                  onClick={() => setShowLangMenu((v) => !v)}
                >
                  <Globe className="h-3.5 w-3.5" />
                  <span>{selectedLang.native}</span>
                </Button>

                {showLangMenu && (
                  <div className="absolute right-0 top-full mt-1 z-50 bg-popover border rounded-lg shadow-lg py-1 min-w-[160px] max-h-72 overflow-y-auto">
                    {LANGUAGES.map((lang) => (
                      <button
                        key={lang.code}
                        className={cn(
                          "w-full px-3 py-2 text-left text-sm hover:bg-accent transition-colors flex items-center justify-between",
                          selectedLang.code === lang.code && "text-primary font-medium"
                        )}
                        onClick={() => {
                          setSelectedLang(lang);
                          setShowLangMenu(false);
                          if (isListening) recognitionRef.current?.stop();
                        }}
                      >
                        <span>{lang.native}</span>
                        <span className="text-xs text-muted-foreground">{lang.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

            </div>
          </CardHeader>

          {/* Messages */}
          <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3 animate-fade-in",
                  message.type === "user" ? "flex-row-reverse" : ""
                )}
              >
                <div
                  className={cn(
                    "h-8 w-8 rounded-full flex items-center justify-center shrink-0",
                    message.type === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary"
                  )}
                >
                  {message.type === "user" ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Sparkles className="h-4 w-4 text-primary" />
                  )}
                </div>

                <div
                  className={cn(
                    "max-w-[80%] rounded-2xl px-4 py-3",
                    message.type === "user"
                      ? "bg-primary text-primary-foreground rounded-tr-md"
                      : "bg-secondary rounded-tl-md"
                  )}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  <p
                    className={cn(
                      "text-xs mt-2",
                      message.type === "user"
                        ? "text-primary-foreground/70"
                        : "text-muted-foreground"
                    )}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {isTyping && (
              <div className="flex gap-3 animate-fade-in">
                <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center">
                  <Sparkles className="h-4 w-4 text-primary" />
                </div>
                <div className="bg-secondary rounded-2xl rounded-tl-md px-4 py-3">
                  <div className="flex items-center gap-1">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </CardContent>

          {/* Voice status bar */}
          {(isListening || voiceError || interimText) && (
            <div
              className={cn(
                "px-4 py-2 text-xs flex items-center gap-2 border-t",
                voiceError
                  ? "bg-destructive/10 text-destructive"
                  : "bg-primary/5 text-primary"
              )}
            >
              {isListening && !voiceError && (
                <>
                  <span className="h-2 w-2 rounded-full bg-primary animate-pulse shrink-0" />
                  <span className="truncate">
                    {interimText
                      ? `Hearing: "${interimText}"`
                      : `Listening in ${selectedLang.native}... Speak now`}
                  </span>
                </>
              )}
              {voiceError && <span>{voiceError}</span>}
            </div>
          )}

          {/* Quick Questions */}
          <div className="px-4 pb-2 pt-2">
            <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
              {quickQuestions.map((question) => (
                <Button
                  key={question}
                  variant="secondary"
                  size="sm"
                  className="whitespace-nowrap text-xs"
                  onClick={() => handleQuickQuestion(question)}
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>

          {/* Input bar */}
          <div className="p-4 border-t">
            <form
              onSubmit={(e) => { e.preventDefault(); sendMessage(); }}
              className="flex gap-2"
            >
              <div className="relative flex-1">
                <Input
                  value={
                    interimText
                      ? (input ? input + " " : "") + interimText
                      : input
                  }
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={
                    isListening
                      ? `Listening in ${selectedLang.native}...`
                      : "Ask me anything about farming..."
                  }
                  className={cn("pr-10", isListening && "border-primary")}
                  readOnly={isListening}
                />

                {/* Mic toggle button */}
                <button
                  type="button"
                  onClick={startListening}
                  title={
                    isListening
                      ? "Stop listening"
                      : `Voice input (${selectedLang.native})`
                  }
                  className={cn(
                    "absolute right-3 top-1/2 -translate-y-1/2 transition-colors",
                    isListening
                      ? "text-destructive animate-pulse"
                      : "text-muted-foreground hover:text-primary"
                  )}
                >
                  {isListening ? (
                    <MicOff className="h-4 w-4" />
                  ) : (
                    <Mic className="h-4 w-4" />
                  )}
                </button>
              </div>

              <Button type="submit" disabled={!input.trim() || isTyping}>
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>

        </Card>
      </div>
    </DashboardLayout>
  );
};

export default Chatbot;