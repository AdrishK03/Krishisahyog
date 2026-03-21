import React, { createContext, useContext, useState, useEffect } from "react";
import type { ReactNode } from "react";
import { authAPI } from "@/services/api";

interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string, name: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const storedUser = localStorage.getItem("krishisahyog_user");
    const token = localStorage.getItem("krishisahyog_token");
    if (storedUser && token) {
      try {
        setUser(JSON.parse(storedUser));
      } catch {
        localStorage.removeItem("krishisahyog_user");
        localStorage.removeItem("krishisahyog_token");
      }
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    setIsLoading(true);
    try {
      const { data } = await authAPI.login(email, password);
      const userData: User = { id: data.user.id, email: data.user.email, name: data.user.name };
      localStorage.setItem("krishisahyog_token", data.access_token);
      localStorage.setItem("krishisahyog_user", JSON.stringify(userData));
      setUser(userData);
      setIsLoading(false);
      return { success: true };
    } catch (err: unknown) {
      setIsLoading(false);
      const detail = (err as { response?: { data?: { detail?: string | { msg?: string }[] } } })?.response?.data?.detail;
      const msg = typeof detail === "string" ? detail : Array.isArray(detail) ? detail[0]?.msg : null;
      return { success: false, error: msg || "Login failed. Please try again." };
    }
  };

  const register = async (email: string, password: string, name: string): Promise<{ success: boolean; error?: string }> => {
    setIsLoading(true);
    try {
      const { data } = await authAPI.register(email, password, name);
      const userData: User = { id: data.user.id, email: data.user.email, name: data.user.name };
      localStorage.setItem("krishisahyog_token", data.access_token);
      localStorage.setItem("krishisahyog_user", JSON.stringify(userData));
      setUser(userData);
      setIsLoading(false);
      return { success: true };
    } catch (err: unknown) {
      setIsLoading(false);
      const detail = (err as { response?: { data?: { detail?: string | { msg?: string }[] } } })?.response?.data?.detail;
      const msg = typeof detail === "string" ? detail : Array.isArray(detail) ? detail[0]?.msg : null;
      return { success: false, error: msg || "Registration failed. Please try again." };
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("krishisahyog_user");
    localStorage.removeItem("krishisahyog_token");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
