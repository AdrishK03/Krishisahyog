import { useState } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mountain, Droplets, Thermometer, Leaf, RefreshCw, TrendingUp, AlertCircle, Sparkles } from "lucide-react";
import { predictionAPI } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

interface SoilMetric {
  label: string;
  value: number;
  unit: string;
  optimal: { min: number; max: number };
  icon: typeof Droplets;
  color: string;
}

const soilMetrics: SoilMetric[] = [
  { label: "pH Level", value: 6.8, unit: "", optimal: { min: 6.0, max: 7.5 }, icon: Droplets, color: "primary" },
  { label: "Moisture", value: 42, unit: "%", optimal: { min: 40, max: 60 }, icon: Droplets, color: "info" },
  { label: "Temperature", value: 24, unit: "°C", optimal: { min: 20, max: 30 }, icon: Thermometer, color: "warning" },
  { label: "Nitrogen (N)", value: 45, unit: "ppm", optimal: { min: 60, max: 120 }, icon: Leaf, color: "success" },
  { label: "Phosphorus (P)", value: 35, unit: "ppm", optimal: { min: 25, max: 50 }, icon: Leaf, color: "accent" },
  { label: "Potassium (K)", value: 180, unit: "ppm", optimal: { min: 150, max: 250 }, icon: Leaf, color: "primary" },
];

const recommendations = [
  {
    type: "warning",
    title: "Nitrogen Deficiency",
    description: "Current nitrogen levels are below optimal. Consider adding nitrogen-rich fertilizers or planting legumes.",
    action: "Add organic compost or use ammonium sulfate",
  },
  {
    type: "success",
    title: "pH Balance Optimal",
    description: "Your soil pH is within the ideal range for most crops. Continue current maintenance practices.",
    action: "Monitor monthly",
  },
  {
    type: "info",
    title: "Moisture Levels Good",
    description: "Current moisture levels are adequate. Adjust irrigation based on weather forecasts.",
    action: "Check weekly during dry spells",
  },
];

interface FertilizerResult {
  recommended_fertilizer: string;
  explanation: string;
  model_used: "real" | "dummy";
}

const SoilAnalysis = () => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [fertilizerResult, setFertilizerResult] = useState<FertilizerResult | null>(null);
  const [isLoadingRec, setIsLoadingRec] = useState(false);
  const { toast } = useToast();

  const refreshData = () => {
    setIsRefreshing(true);
    setFertilizerResult(null);
    setTimeout(() => setIsRefreshing(false), 2000);
  };

  const getFertilizerRecommendation = async () => {
    setIsLoadingRec(true);
    setFertilizerResult(null);
    try {
      const n = soilMetrics.find((m) => m.label.includes("Nitrogen"))?.value ?? 45;
      const p = soilMetrics.find((m) => m.label.includes("Phosphorus"))?.value ?? 35;
      const k = soilMetrics.find((m) => m.label.includes("Potassium"))?.value ?? 180;
      const ph = soilMetrics.find((m) => m.label.includes("pH"))?.value ?? 6.8;
      const moisture = soilMetrics.find((m) => m.label.includes("Moisture"))?.value ?? 42;
      const temp = soilMetrics.find((m) => m.label.includes("Temperature"))?.value ?? 24;
      const { data } = await predictionAPI.soilFertilizer({
        nitrogen: n,
        phosphorus: p,
        potassium: k,
        ph,
        moisture,
        temperature: temp,
      });
      setFertilizerResult(data);
      toast({
        title: "Recommendation Ready",
        description: data.model_used === "dummy"
          ? "Rule-based demo. Add trained model for real predictions."
          : "ML-based fertilizer recommendation.",
      });
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Request failed.";
      toast({ title: "Error", description: msg, variant: "destructive" });
    } finally {
      setIsLoadingRec(false);
    }
  };

  const getStatus = (value: number, optimal: { min: number; max: number }) => {
    if (value < optimal.min) return { status: "low", color: "bg-warning" };
    if (value > optimal.max) return { status: "high", color: "bg-destructive" };
    return { status: "optimal", color: "bg-success" };
  };

  const getProgressColor = (value: number, optimal: { min: number; max: number }) => {
    if (value < optimal.min) return "bg-warning";
    if (value > optimal.max) return "bg-destructive";
    return "bg-success";
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
              <Mountain className="h-8 w-8 text-primary" />
              Soil Analysis
            </h1>
            <p className="text-muted-foreground mt-1">
              Real-time soil monitoring and nutrient analysis
            </p>
          </div>
          <Button variant="outline" onClick={refreshData} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            {isRefreshing ? "Refreshing..." : "Refresh Data"}
          </Button>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {soilMetrics.map((metric) => {
            const { status } = getStatus(metric.value, metric.optimal);
            const progressValue = (metric.value / metric.optimal.max) * 100;
            
            return (
              <Card key={metric.label} className="overflow-hidden">
                <CardContent className="p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-2.5 rounded-lg bg-${metric.color}/10`}>
                      <metric.icon className={`h-5 w-5 text-${metric.color}`} />
                    </div>
                    <span className={`text-xs font-medium px-2 py-1 rounded-full capitalize ${
                      status === "optimal" ? "bg-success/10 text-success" :
                      status === "low" ? "bg-warning/10 text-warning" :
                      "bg-destructive/10 text-destructive"
                    }`}>
                      {status}
                    </span>
                  </div>
                  
                  <h3 className="text-sm text-muted-foreground mb-1">{metric.label}</h3>
                  <p className="text-2xl font-bold mb-3">
                    {metric.value}{metric.unit}
                  </p>
                  
                  <div className="space-y-2">
                    <div className="h-2 rounded-full bg-secondary overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${getProgressColor(metric.value, metric.optimal)}`}
                        style={{ width: `${Math.min(progressValue, 100)}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Optimal: {metric.optimal.min} - {metric.optimal.max}{metric.unit}
                    </p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Fertilizer Recommendation */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Fertilizer Recommendation
            </CardTitle>
            <CardDescription>
              Get ML-powered fertilizer advice based on your soil parameters above
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!fertilizerResult ? (
              <Button
                onClick={getFertilizerRecommendation}
                disabled={isLoadingRec}
                className="w-full sm:w-auto"
              >
                {isLoadingRec ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Analyzing...
                  </span>
                ) : (
                  <>Get Recommendation</>
                )}
              </Button>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold">{fertilizerResult.recommended_fertilizer}</span>
                  {fertilizerResult.model_used === "dummy" && (
                    <span className="text-xs px-2 py-1 rounded-full bg-warning/20 text-warning">
                      Demo mode
                    </span>
                  )}
                </div>
                <p className="text-sm text-muted-foreground">{fertilizerResult.explanation}</p>
                <Button variant="outline" size="sm" onClick={getFertilizerRecommendation} disabled={isLoadingRec}>
                  Refresh
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recommendations */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              General Recommendations
            </CardTitle>
            <CardDescription>
              Suggestions based on your soil analysis
            </CardDescription>
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
                    <AlertCircle className={`h-5 w-5 mt-0.5 ${
                      rec.type === "warning" ? "text-warning" :
                      rec.type === "success" ? "text-success" : "text-info"
                    }`} />
                    <div className="flex-1">
                      <h4 className="font-semibold mb-1">{rec.title}</h4>
                      <p className="text-sm text-muted-foreground mb-2">{rec.description}</p>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-primary">Recommended Action:</span>
                        <span className="text-xs text-muted-foreground">{rec.action}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Soil Health Score */}
        <Card className="bg-gradient-to-br from-primary/5 to-accent/5">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Overall Soil Health Score</h3>
                <p className="text-muted-foreground text-sm max-w-md">
                  Based on all soil parameters, your field has a good health score. Focus on improving nitrogen levels for better yields.
                </p>
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
                      strokeDasharray={`${78 * 2.51} ${100 * 2.51}`}
                      className="text-primary"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl font-bold">78%</span>
                  </div>
                </div>
                <div className="text-left">
                  <p className="text-sm text-muted-foreground">Status</p>
                  <p className="text-lg font-semibold text-success">Good</p>
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
