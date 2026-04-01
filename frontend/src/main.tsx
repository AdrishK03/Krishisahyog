import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

const LANGUAGE_STORAGE_KEY = "krishisahyog_language";

const setTranslateCookie = (langCode: string) => {
  document.cookie = `googtrans=/en/${langCode}; path=/`;
  document.cookie = `googtrans=/en/${langCode}; path=/; domain=${window.location.hostname}`;
};

const safeLang = localStorage.getItem(LANGUAGE_STORAGE_KEY) ?? "en";
setTranslateCookie(safeLang);

// Google Translate mutates DOM nodes and can throw during React updates.
// Guard these operations to prevent full-page crashes in translated mode.
const originalRemoveChild = Node.prototype.removeChild;
Node.prototype.removeChild = function removeChildSafe(child: Node) {
  try {
    return originalRemoveChild.call(this, child);
  } catch (error) {
    if (error instanceof DOMException) return child;
    throw error;
  }
};

const originalInsertBefore = Node.prototype.insertBefore;
Node.prototype.insertBefore = function insertBeforeSafe(newNode: Node, referenceNode: Node | null) {
  try {
    return originalInsertBefore.call(this, newNode, referenceNode);
  } catch (error) {
    if (error instanceof DOMException) return newNode;
    throw error;
  }
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
