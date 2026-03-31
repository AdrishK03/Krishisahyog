import { useEffect, useState } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Settings as SettingsIcon, Bell, Globe, Palette, Shield } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";

type LanguageCode = "en" | "hi" | "bn" | "pa" | "ta";

const Settings = () => {
  const { user } = useAuth();
  const { toast } = useToast();

  const [language, setLanguage] = useState<LanguageCode>("en");
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [weatherAlerts, setWeatherAlerts] = useState(true);
  const [diagnosisHistory, setDiagnosisHistory] = useState(true);
  const [compactMode, setCompactMode] = useState(false);

  useEffect(() => {
    const savedLanguage = (localStorage.getItem("krishisahyog_language") as LanguageCode | null) ?? "en";
    const savedEmailAlerts = localStorage.getItem("krishisahyog_email_alerts");
    const savedWeatherAlerts = localStorage.getItem("krishisahyog_weather_alerts");
    const savedDiagnosisHistory = localStorage.getItem("krishisahyog_diagnosis_history");
    const savedCompactMode = localStorage.getItem("krishisahyog_compact_mode");

    setLanguage(savedLanguage);
    setEmailAlerts(savedEmailAlerts !== "false");
    setWeatherAlerts(savedWeatherAlerts !== "false");
    setDiagnosisHistory(savedDiagnosisHistory !== "false");
    setCompactMode(savedCompactMode === "true");
  }, []);

  const saveSettings = () => {
    localStorage.setItem("krishisahyog_language", language);
    localStorage.setItem("krishisahyog_email_alerts", String(emailAlerts));
    localStorage.setItem("krishisahyog_weather_alerts", String(weatherAlerts));
    localStorage.setItem("krishisahyog_diagnosis_history", String(diagnosisHistory));
    localStorage.setItem("krishisahyog_compact_mode", String(compactMode));

    toast({
      title: "Settings Saved",
      description: "Your preferences have been updated.",
    });
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <SettingsIcon className="h-8 w-8 text-primary" />
            Settings
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage language, alerts, and dashboard preferences
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5 text-primary" />
                Language
              </CardTitle>
              <CardDescription>
                Set your preferred app language. English is most stable for ML pages.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select value={language} onValueChange={(v) => setLanguage(v as LanguageCode)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="en">English</SelectItem>
                  <SelectItem value="hi">Hindi (हिंदी)</SelectItem>
                  <SelectItem value="bn">Bengali (বাংলা)</SelectItem>
                  <SelectItem value="pa">Punjabi (ਪੰਜਾਬੀ)</SelectItem>
                  <SelectItem value="ta">Tamil (தமிழ்)</SelectItem>
                </SelectContent>
              </Select>
              <Badge variant="outline">Preferred language: {language.toUpperCase()}</Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5 text-primary" />
                Notifications
              </CardTitle>
              <CardDescription>
                Choose what alerts you want to receive.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Email Alerts</span>
                <Switch checked={emailAlerts} onCheckedChange={setEmailAlerts} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Weather Alerts</span>
                <Switch checked={weatherAlerts} onCheckedChange={setWeatherAlerts} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Save Diagnosis History</span>
                <Switch checked={diagnosisHistory} onCheckedChange={setDiagnosisHistory} />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Palette className="h-5 w-5 text-primary" />
                Appearance
              </CardTitle>
              <CardDescription>
                Adjust dashboard readability preferences.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-sm">Compact Dashboard Cards</span>
                <Switch checked={compactMode} onCheckedChange={setCompactMode} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                Account
              </CardTitle>
              <CardDescription>
                Basic account profile details.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p><span className="text-muted-foreground">Name:</span> {user?.name ?? "Farmer"}</p>
              <p><span className="text-muted-foreground">Email:</span> {user?.email ?? "Not available"}</p>
            </CardContent>
          </Card>
        </div>

        <div className="flex justify-end">
          <Button onClick={saveSettings}>Save Settings</Button>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Settings;
