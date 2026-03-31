import { useEffect, useState } from "react";
import { Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

declare global {
  interface Window {
    google: any;
    googleTranslateElementInit: () => void;
  }
}

const languages = [
  { code: "en", name: "English", native: "English" },
  { code: "hi", name: "Hindi", native: "हिंदी" },
  { code: "bn", name: "Bengali", native: "বাংলা" },
  { code: "pa", name: "Punjabi", native: "ਪੰਜਾਬੀ" },
  { code: "ta", name: "Tamil", native: "தமிழ்" },
];

const GoogleTranslate = () => {
  const [currentLang, setCurrentLang] = useState("en");
  const [, setIsLoaded] = useState(false);

  useEffect(() => {
    // Add Google Translate script
    const addScript = () => {
      if (document.getElementById("google-translate-script")) {
        setIsLoaded(true);
        return;
      }

      // Create hidden element for Google Translate
      const translateDiv = document.createElement("div");
      translateDiv.id = "google_translate_element";
      translateDiv.style.display = "none";
      document.body.appendChild(translateDiv);

      // Initialize Google Translate
      window.googleTranslateElementInit = () => {
        new window.google.translate.TranslateElement(
          {
            pageLanguage: "en",
            includedLanguages: "en,hi,bn,pa,ta",
            autoDisplay: false,
          },
          "google_translate_element"
        );
        setIsLoaded(true);
      };

      // Add script
      const script = document.createElement("script");
      script.id = "google-translate-script";
      script.src = "//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
      script.async = true;
      document.body.appendChild(script);
    };

    addScript();

    // Hide Google Translate bar
    const style = document.createElement("style");
    style.innerHTML = `
      .goog-te-banner-frame,
      .goog-te-balloon-frame,
      #goog-gt-tt { display: none !important; }
      .goog-te-menu-value:hover { text-decoration: none !important; }
      body { top: 0 !important; }
      .goog-logo-link { display: none !important; }
      .goog-te-gadget { color: transparent !important; }
      .goog-te-gadget .goog-te-combo { color: initial; }
      .goog-text-highlight { background: none !important; box-shadow: none !important; }
    `;
    document.head.appendChild(style);

    return () => {
      style.remove();
    };
  }, []);

  const changeLanguage = (langCode: string) => {
    setCurrentLang(langCode);
    
    // Trigger Google Translate language change
    const selectElement = document.querySelector(".goog-te-combo") as HTMLSelectElement;
    if (selectElement) {
      selectElement.value = langCode;
      selectElement.dispatchEvent(new Event("change"));
      // React UIs can break when Google Translate mutates the DOM live.
      // Reload keeps page and virtual DOM in sync after language switch.
      window.setTimeout(() => window.location.reload(), 300);
    } else {
      // Fallback: Set cookie and reload
      document.cookie = `googtrans=/en/${langCode}; path=/`;
      document.cookie = `googtrans=/en/${langCode}; path=/; domain=${window.location.hostname}`;
      window.location.reload();
    }
  };

  const currentLanguage = languages.find(l => l.code === currentLang) || languages[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-2">
          <Globe className="h-4 w-4" />
          <span className="hidden sm:inline">{currentLanguage.native}</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        {languages.map((lang) => (
          <DropdownMenuItem
            key={lang.code}
            onClick={() => changeLanguage(lang.code)}
            className={`flex justify-between ${currentLang === lang.code ? "bg-primary/10" : ""}`}
          >
            <span>{lang.name}</span>
            <span className="text-muted-foreground">{lang.native}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default GoogleTranslate;
