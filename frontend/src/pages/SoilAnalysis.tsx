import { useState, useEffect, useMemo } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Mountain, Droplets, Thermometer, Leaf, RefreshCw, TrendingUp, AlertCircle, Sparkles, Wifi, WifiOff } from "lucide-react";
import { predictionAPI } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import {
  subscribePlantSensor,
  emptySensorState,
  isFirebaseSoilConfigured,
  type PlantSensorState,
  type FirebaseSoilStatus,
} from "@/lib/firebaseSoil";

const SOIL_TYPES = ["Red", "Black", "Sandy", "Loamy", "Clayey"] as const;
const CROP_TYPES = [
  "Barley",
  "Cotton",
  "Ground Nuts",
  "Maize",
  "Millets",
  "Oil seeds",
  "Paddy",
  "Pulses",
  "Sugarcane",
  "Tobacco",
  "Wheat",
] as const;

type MetricRow = {
  key: keyof PlantSensorState;
  label: string;
  unit: string;
  optimal: { min: number; max: number };
  icon: typeof Droplets;
  iconClass: string;
};

const METRICS: MetricRow[] = [
  { key: "moisture", label: "Moisture", unit: "%", optimal: { min: 40, max: 60 }, icon: Droplets, iconClass: "bg-sky-500/10 text-sky-600" },
  { key: "humidity", label: "Humidity", unit: "%", optimal: { min: 40, max: 70 }, icon: Droplets, iconClass: "bg-cyan-500/10 text-cyan-600" },
  { key: "temperature", label: "Temperature", unit: "°C", optimal: { min: 20, max: 30 }, icon: Thermometer, iconClass: "bg-amber-500/10 text-amber-600" },
  { key: "nitrogen", label: "Nitrogen (N)", unit: " ppm", optimal: { min: 60, max: 120 }, icon: Leaf, iconClass: "bg-emerald-500/10 text-emerald-600" },
  { key: "phosphorus", label: "Phosphorus (P)", unit: " ppm", optimal: { min: 25, max: 50 }, icon: Leaf, iconClass: "bg-violet-500/10 text-violet-600" },
  { key: "potassium", label: "Potassium (K)", unit: " ppm", optimal: { min: 150, max: 250 }, icon: Leaf, iconClass: "bg-green-600/10 text-green-700" },
];

interface FertilizerResult {
  recommended_fertilizer: string;
  explanation: string;
  model_used: "real" | "dummy";
}

function formatMetricValue(v: number | null, unit: string): string {
  if (v == null || Number.isNaN(v)) return "—";
  const rounded = Math.abs(v) >= 100 ? Math.round(v * 10) / 10 : Math.round(v * 100) / 100;
  return `${rounded}${unit}`;
}

function getStatus(value: number | null, optimal: { min: number; max: number }) {
  if (value == null || Number.isNaN(value)) return { status: "pending" as const, color: "bg-muted" };
  if (value < optimal.min) return { status: "low" as const, color: "bg-warning" };
  if (value > optimal.max) return { status: "high" as const, color: "bg-destructive" };
  return { status: "optimal" as const, color: "bg-success" };
}

function getProgressColor(value: number | null, optimal: { min: number; max: number }) {
  const { color } = getStatus(value, optimal);
  return color;
}

function computeHealth(sensor: PlantSensorState): { pct: number; label: string; detail: string } {
  let inRange = 0;
  let total = 0;
  for (const m of METRICS) {
    const v = sensor[m.key];
    if (v == null || Number.isNaN(v)) continue;
    total += 1;
    if (v >= m.optimal.min && v <= m.optimal.max) inRange += 1;
  }
  if (total === 0) {
    return {
      pct: 0,
      label: "Waiting",
      detail: "Connect Firebase or wait for sensor data to compute a health score.",
    };
  }
  const pct = Math.round((inRange / total) * 100);
  const label = pct >= 70 ? "Good" : pct >= 40 ? "Fair" : "Needs attention";
  return {
    pct,
    label,
    detail: `${inRange} of ${total} reported readings are within the suggested optimal bands.`,
  };
}

const SoilAnalysis = () => {
  const [sensor, setSensor] = useState<PlantSensorState>(emptySensorState);
  const [fbStatus, setFbStatus] = useState<FirebaseSoilStatus>("connecting");
  const [fbMessage, setFbMessage] = useState<string>("");
  const [soilType, setSoilType] = useState<string>(SOIL_TYPES[0]);
  const [cropType, setCropType] = useState<string>(CROP_TYPES[0]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [fertilizerResult, setFertilizerResult] = useState<FertilizerResult | null>(null);
  const [isLoadingRec, setIsLoadingRec] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    const unsub = subscribePlantSensor(
      setSensor,
      (status, msg) => {
        setFbStatus(status);
        setFbMessage(msg ?? "");
      }
    );
    return () => unsub();
  }, []);

  const health = useMemo(() => computeHealth(sensor), [sensor]);

  const recommendations = useMemo(() => {
    const items: { type: "warning" | "success" | "info"; title: string; description: string; action: string }[] = [];
    if (sensor.nitrogen != null && sensor.nitrogen < 60) {
      items.push({
        type: "warning",
        title: "Nitrogen below range",
        description: "Live N reading is under the typical optimal band. Consider nitrogen supplementation suited to your crop.",
        action: "Add compost or consult an agronomist for N dose",
      });
    }
    if (sensor.ph != null) {
      if (sensor.ph >= 6 && sensor.ph <= 7.5) {
        items.push({
          type: "success",
          title: "pH in a favorable band",
          description: "Reported pH suits many common crops. Keep monitoring as fertilizers and irrigation shift pH over time.",
          action: "Test monthly or after major amendments",
        });
      } else {
        items.push({
          type: "warning",
          title: "pH outside common optimal range",
          description: "Adjust liming or acidification based on soil tests and crop needs.",
          action: "Soil test and follow local extension guidance",
        });
      }
    }
    if (sensor.moisture != null) {
      items.push({
        type: sensor.moisture >= 40 && sensor.moisture <= 60 ? "success" : "info",
        title:
          sensor.moisture >= 40 && sensor.moisture <= 60
            ? "Moisture in a comfortable range"
            : "Review irrigation",
        description:
          sensor.moisture >= 40 && sensor.moisture <= 60
            ? "Moisture looks adequate for many field conditions; tune for your crop stage."
            : "Moisture is outside the typical 40–60% window used on this dashboard.",
        action: sensor.moisture < 40 ? "Increase irrigation if wilting risk" : "Improve drainage or reduce waterlogging risk",
      });
    }
    if (items.length === 0) {
      return [
        {
          type: "info" as const,
          title: "Waiting for readings",
          description: "Once Firebase sends NPK, moisture, and pH, tailored tips will appear here.",
          action: "Ensure the device writes to the plant/ path in Realtime Database",
        },
      ];
    }
    return items;
  }, [sensor]);

  const refreshData = () => {
    setIsRefreshing(true);
    setFertilizerResult(null);
    setTimeout(() => setIsRefreshing(false), 800);
  };

  const getFertilizerRecommendation = async () => {
    const t = sensor.temperature;
    const h = sensor.humidity;
    const m = sensor.moisture;
    const n = sensor.nitrogen;
    const p = sensor.phosphorus;
    const k = sensor.potassium;
    const missing: string[] = [];
    if (t == null || Number.isNaN(t)) missing.push("Temperature");
    if (h == null || Number.isNaN(h)) missing.push("Humidity");
    if (m == null || Number.isNaN(m)) missing.push("Moisture");
    if (n == null || Number.isNaN(n)) missing.push("Nitrogen");
    if (p == null || Number.isNaN(p)) missing.push("Phosphorous");
    if (k == null || Number.isNaN(k)) missing.push("Potassium");
    if (missing.length) {
      toast({
        title: "Missing sensor data",
        description: `Need numeric values for: ${missing.join(", ")}`,
        variant: "destructive",
      });
      return;
    }

    setIsLoadingRec(true);
    setFertilizerResult(null);
    try {
      const { data } = await predictionAPI.soilFertilizer({
        Temperature: t!,
        Humidity: h!,
        Moisture: m!,
        Soil_Type: soilType,
        Crop_Type: cropType,
        Nitrogen: n!,
        Phosphorous: p!,
        Potassium: k!,
      });
      setFertilizerResult(data);
      toast({
        title: "Recommendation ready",
        description:
          data.model_used === "dummy"
            ? "Rule-based demo output."
            : "ML-based fertilizer recommendation from your live readings.",
      });
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Request failed.";
      toast({ title: "Error", description: msg, variant: "destructive" });
    } finally {
      setIsLoadingRec(false);
    }
  };

  const statusBannerClass =
    fbStatus === "live"
      ? "bg-emerald-500/10 text-emerald-800 dark:text-emerald-200 border-emerald-500/30"
      : fbStatus === "error" || fbStatus === "no_data"
        ? "bg-destructive/10 text-destructive border-destructive/30"
        : fbStatus === "unconfigured"
          ? "bg-amber-500/10 text-amber-900 dark:text-amber-200 border-amber-500/30"
          : "bg-muted text-muted-foreground border-border";

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
              <Mountain className="h-8 w-8 text-primary" />
              Soil Analysis
            </h1>
            <p className="text-muted-foreground mt-1">Real-time soil monitoring and nutrient analysis</p>
          </div>
          <Button variant="outline" onClick={refreshData} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            {isRefreshing ? "Refreshing..." : "Refresh Data"}
          </Button>
        </div>

        <div
          className={`flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg border ${statusBannerClass}`}
        >
          {fbStatus === "live" ? <Wifi className="h-4 w-4 shrink-0" /> : <WifiOff className="h-4 w-4 shrink-0" />}
          <span>
            {fbStatus === "unconfigured" && "Firebase not configured — set VITE_FIREBASE_* env vars for live data."}
            {fbStatus === "connecting" && "Connecting to Firebase…"}
            {fbStatus === "live" && (fbMessage || "Live")}
            {fbStatus === "no_data" && (fbMessage || "No data at plant/")}
            {fbStatus === "error" && `Firebase: ${fbMessage || "error"}`}
          </span>
        </div>

        {!isFirebaseSoilConfigured() && (
          <p className="text-sm text-muted-foreground">
            Without Firebase, metrics stay empty. The ML recommendation still needs all sensor numbers from your database
            path <code className="text-xs bg-muted px-1 rounded">plant/</code>.
          </p>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Soil type</CardTitle>
              <CardDescription>Choose the soil class (must match model training labels)</CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={soilType} onValueChange={setSoilType}>
                <SelectTrigger>
                  <SelectValue placeholder="Soil type" />
                </SelectTrigger>
                <SelectContent>
                  {SOIL_TYPES.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Target crop</CardTitle>
              <CardDescription>Crop you are planning for this reading</CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={cropType} onValueChange={setCropType}>
                <SelectTrigger>
                  <SelectValue placeholder="Crop" />
                </SelectTrigger>
                <SelectContent>
                  {CROP_TYPES.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {METRICS.map((metric) => {
            const value = sensor[metric.key];
            const { status } = getStatus(value, metric.optimal);
            const progressValue =
              value != null && !Number.isNaN(value) ? (value / metric.optimal.max) * 100 : 0;
            const Icon = metric.icon;

            return (
              <Card key={metric.key} className="overflow-hidden">
                <CardContent className="p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-2.5 rounded-lg ${metric.iconClass}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <span
                      className={`text-xs font-medium px-2 py-1 rounded-full capitalize ${
                        status === "optimal"
                          ? "bg-success/10 text-success"
                          : status === "low" || status === "high"
                            ? status === "low"
                              ? "bg-warning/10 text-warning"
                              : "bg-destructive/10 text-destructive"
                            : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {status}
                    </span>
                  </div>

                  <h3 className="text-sm text-muted-foreground mb-1">{metric.label}</h3>
                  <p className="text-2xl font-bold mb-3">{formatMetricValue(value, metric.unit)}</p>

                  <div className="space-y-2">
                    <div className="h-2 rounded-full bg-secondary overflow-hidden">
                      {value != null && !Number.isNaN(value) ? (
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${getProgressColor(value, metric.optimal)}`}
                          style={{ width: `${Math.min(progressValue, 100)}%` }}
                        />
                      ) : null}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Optimal: {metric.optimal.min} - {metric.optimal.max}
                      {metric.unit}
                    </p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Fertilizer recommendation
            </CardTitle>
            <CardDescription>
              Uses live Temperature, Humidity, Moisture, N, P, K plus your soil and crop selections
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!fertilizerResult ? (
              <Button onClick={getFertilizerRecommendation} disabled={isLoadingRec} className="w-full sm:w-auto">
                {isLoadingRec ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Analyzing…
                  </span>
                ) : (
                  <>Get recommendation</>
                )}
              </Button>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-lg">{fertilizerResult.recommended_fertilizer}</span>
                  {fertilizerResult.model_used === "dummy" && (
                    <span className="text-xs px-2 py-1 rounded-full bg-warning/20 text-warning">Demo mode</span>
                  )}
                </div>
                <p className="text-sm text-muted-foreground">{fertilizerResult.explanation}</p>
                <Button variant="outline" size="sm" onClick={getFertilizerRecommendation} disabled={isLoadingRec}>
                  Run again
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              General recommendations
            </CardTitle>
            <CardDescription>Suggestions from current readings</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recommendations.map((rec, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-xl border ${
                    rec.type === "warning"
                      ? "bg-warning/5 border-warning/20"
                      : rec.type === "success"
                        ? "bg-success/5 border-success/20"
                        : "bg-info/5 border-info/20"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <AlertCircle
                      className={`h-5 w-5 mt-0.5 shrink-0 ${
                        rec.type === "warning"
                          ? "text-warning"
                          : rec.type === "success"
                            ? "text-success"
                            : "text-info"
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <h4 className="font-semibold mb-1">{rec.title}</h4>
                      <p className="text-sm text-muted-foreground mb-2">{rec.description}</p>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-primary">Suggested action:</span>
                        <span className="text-xs text-muted-foreground">{rec.action}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-primary/5 to-accent/5">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Overall soil health score</h3>
                <p className="text-muted-foreground text-sm max-w-md">{health.detail}</p>
              </div>
              <div className="flex items-center gap-4">
                <div className="relative h-32 w-32">
                  <svg className="transform -rotate-90" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="8"
                      className="text-secondary"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="8"
                      strokeDasharray={`${health.pct * 2.51} ${100 * 2.51}`}
                      className="text-primary"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl font-bold">{health.pct}%</span>
                  </div>
                </div>
                <div className="text-left">
                  <p className="text-sm text-muted-foreground">Status</p>
                  <p className="text-lg font-semibold text-success">{health.label}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default SoilAnalysis;
