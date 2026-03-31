import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import ProtectedRoute from "@/components/auth/ProtectedRoute";

// Pages
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import Diagnosis from "./pages/Diagnosis";
import SoilAnalysis from "./pages/SoilAnalysis";
import Weather from "./pages/Weather";
import Chatbot from "./pages/Chatbot";
import Market from "./pages/Market";
import Features from "./pages/Features";
import Advisory from "./pages/Advisory";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/features" element={<Features />} />
            
            {/* Protected Dashboard Routes */}
            <Route path="/dashboard" element={
              <ProtectedRoute><Dashboard /></ProtectedRoute>
            } />
            <Route path="/dashboard/diagnosis" element={
              <ProtectedRoute><Diagnosis /></ProtectedRoute>
            } />
            <Route path="/dashboard/soil" element={
              <ProtectedRoute><SoilAnalysis /></ProtectedRoute>
            } />
            <Route path="/dashboard/weather" element={
              <ProtectedRoute><Weather /></ProtectedRoute>
            } />
            <Route path="/dashboard/chatbot" element={
              <ProtectedRoute><Chatbot /></ProtectedRoute>
            } />
            <Route path="/dashboard/market" element={
              <ProtectedRoute><Market /></ProtectedRoute>
            } />
            <Route path="/dashboard/advisory" element={
              <ProtectedRoute><Advisory /></ProtectedRoute>
            } />
            <Route path="/dashboard/settings" element={
              <ProtectedRoute><Settings /></ProtectedRoute>
            } />
            
            {/* Catch-all */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
