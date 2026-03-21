import { useState, useRef, useEffect } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageCircle, Send, Bot, User, Sparkles, Mic, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { chatAPI } from "@/services/api";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

const quickQuestions = [
  "What crops should I plant this season?",
  "How to treat leaf blight in tomatoes?",
  "Best fertilizer for wheat crops",
  "When to irrigate rice fields?",
];

const DEFAULT_GREETING = "I'm KrishiSahyog AI, your agriculture assistant! I can help with crop recommendations, disease treatment, soil management, and more. What would you like to know?";

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", type: "bot", content: DEFAULT_GREETING, timestamp: new Date() },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    const query = input;
    setInput("");
    setIsTyping(true);

    try {
      const history = messages
        .filter((m) => m.type === "user" || m.type === "bot")
        .map((m) => ({ role: m.type === "user" ? "user" : "assistant", content: m.content }));
      const { data } = await chatAPI.send(query, history);
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: data.response || "Sorry, I couldn't generate a response. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Chat failed.";
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

  return (
    <DashboardLayout>
      <div className="h-[calc(100vh-8rem)] flex flex-col">
        {/* Header */}
        <div className="mb-4">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <MessageCircle className="h-8 w-8 text-primary" />
            Smart Assistant
          </h1>
          <p className="text-muted-foreground mt-1">
            Get personalized farming advice in your language
          </p>
        </div>

        {/* Chat Container */}
        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardHeader className="border-b py-3">
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

          {/* Quick Questions */}
          <div className="px-4 pb-2">
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

          {/* Input */}
          <div className="p-4 border-t">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                sendMessage();
              }}
              className="flex gap-2"
            >
              <div className="relative flex-1">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask me anything about farming..."
                  className="pr-10"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Mic className="h-4 w-4" />
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
