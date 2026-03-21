import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 60000,
});

// Attach JWT from localStorage to every request
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("krishisahyog_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// On 401, clear auth and redirect to login
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("krishisahyog_token");
      localStorage.removeItem("krishisahyog_user");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (email: string, password: string) =>
    api.post<{ access_token: string; user: { id: string; email: string; name: string } }>("/auth/login", { email, password }),

  register: (email: string, password: string, name: string) =>
    api.post<{ access_token: string; user: { id: string; email: string; name: string } }>("/auth/register", { email, password, name }),
};

// ML Prediction API
export const predictionAPI = {
  plantDisease: (imageFile: File) => {
    const formData = new FormData();
    formData.append("file", imageFile);
    return api.post<{
      prediction: string;
      confidence: number;
      model_used: "real" | "dummy";
      disease: string;
      severity: string;
      treatment: string[];
      prevention: string[];
    }>("/predict/plant-disease", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    });
  },

  soilFertilizer: (data: {
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    ph: number;
    moisture: number;
    temperature: number;
  }) =>
    api.post<{
      recommended_fertilizer: string;
      explanation: string;
      model_used: "real" | "dummy";
    }>("/predict/soil-fertilizer", data, { timeout: 30000 }),
};

// Chat API
export const chatAPI = {
  send: (message: string, history?: { role: string; content: string }[]) =>
    api.post<{ response: string; provider?: string }>("/chat", { message, history }, { timeout: 60000 }),
};

export default api;
