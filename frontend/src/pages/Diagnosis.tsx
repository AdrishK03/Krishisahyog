import { useState, useCallback } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Leaf, Upload, Camera, X, CheckCircle, AlertTriangle, Info } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { predictionAPI } from "@/services/api";

interface DiagnosisResult {
  plant: string;
  plantConfidence: number;
  disease: string;
  confidence: number;
  severity: "low" | "medium" | "high";
  treatment: string[];
  prevention: string[];
  model_used?: "real" | "dummy";
}

const Diagnosis = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const { toast } = useToast();

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      processImage(file);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processImage(file);
    }
  };

  const processImage = (file: File) => {
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      setResult(null);
    };
    reader.readAsDataURL(file);
  };

  const analyzePlant = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setResult(null);
    try {
      const { data } = await predictionAPI.plantDisease(selectedFile);
      const plantData = (data?.plant ?? {}) as {
        name?: string;
        confidence?: number;
      };

      const plantConfRaw = Number(plantData.confidence ?? 0);
      const plantConfidence = plantConfRaw <= 1 ? plantConfRaw * 100 : plantConfRaw;

      const diseaseConfidenceRaw = Number(data.confidence ?? 0);
      const diseaseConfidence = diseaseConfidenceRaw <= 1 ? diseaseConfidenceRaw * 100 : diseaseConfidenceRaw;

      setResult({
        plant: plantData.name ?? "Unknown",
        disease: String(data.disease ?? data.prediction ?? "Unknown"),
        confidence: Number.isFinite(diseaseConfidence) ? diseaseConfidence : 0,
        severity: (data.severity as "low" | "medium" | "high") ?? "medium",
        treatment: Array.isArray(data.treatment) ? data.treatment : [],
        prevention: Array.isArray(data.prevention) ? data.prevention : [],
        model_used: data.model_used,
        plantConfidence: Number.isFinite(plantConfidence) ? plantConfidence : 0,
      });
      toast({
        title: "Analysis Complete",
        description: data.model_used === "dummy"
          ? "Demo mode: Add a trained model for real predictions."
          : "Plant health diagnosis is ready.",
      });
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Analysis failed.";
      toast({ title: "Error", description: msg, variant: "destructive" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setResult(null);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "low": return "text-success bg-success/10";
      case "medium": return "text-warning bg-warning/10";
      case "high": return "text-destructive bg-destructive/10";
      default: return "text-muted-foreground bg-muted";
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <Leaf className="h-8 w-8 text-primary" />
            Plant Health Diagnosis
          </h1>
          <p className="text-muted-foreground mt-1">
            Upload a plant image for AI-powered disease detection and treatment recommendations
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Plant Image</CardTitle>
              <CardDescription>
                Take a clear photo of the affected plant leaves or stems
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedImage ? (
                <div
                  className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                    isDragging
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50 hover:bg-secondary/30"
                  }`}
                  onDrop={handleDrop}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                >
                  <div className="flex flex-col items-center gap-4">
                    <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                      <Upload className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">Drag and drop your image here</p>
                      <p className="text-sm text-muted-foreground mt-1">or click to browse files</p>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      onChange={handleFileSelect}
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl overflow-hidden">
                    <img
                      src={selectedImage}
                      alt="Selected plant"
                      className="w-full h-64 object-cover"
                    />
                    <button
                      onClick={clearImage}
                      className="absolute top-3 right-3 h-8 w-8 rounded-full bg-background/80 backdrop-blur-sm flex items-center justify-center hover:bg-background transition-colors"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  <Button
                    onClick={analyzePlant}
                    disabled={isAnalyzing}
                    className="w-full"
                    size="lg"
                  >
                    {isAnalyzing ? (
                      <span className="flex items-center gap-2">
                        <span className="h-4 w-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                        Analysing your plant... (first request may take 15-20 seconds)
                      </span>
                    ) : (
                      <>
                        <Camera className="h-5 w-5 mr-2" />
                        Diagnose Plant
                      </>
                    )}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle>Diagnosis Results</CardTitle>
              <CardDescription>
                AI-powered analysis and recommendations
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!result ? (
                <div className="h-64 flex flex-col items-center justify-center text-center">
                  <div className="h-16 w-16 rounded-full bg-secondary flex items-center justify-center mb-4">
                    <Info className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="text-muted-foreground">
                    Upload and analyze a plant image to see diagnosis results here
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Plant Info */}
                  <div className="flex items-start justify-between gap-4 p-4 rounded-xl bg-secondary/50">
                    <div>
                      <p className="text-sm text-muted-foreground">Detected Plant</p>
                      <p className="text-xl font-bold mt-1">{result.plant}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">Plant Confidence</p>
                      <p className="text-xl font-bold text-primary mt-1">{result.plantConfidence.toFixed(1)}%</p>
                    </div>
                  </div>

                  {/* Disease Info */}
                  <div className="flex items-start justify-between gap-4 p-4 rounded-xl bg-secondary/50">
                    <div>
                      <p className="text-sm text-muted-foreground">Detected Condition</p>
                      <p className="text-xl font-bold mt-1">{result.disease}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">Confidence</p>
                      <p className="text-xl font-bold text-primary mt-1">{result.confidence}%</p>
                    </div>
                  </div>

                  {/* Severity & Model */}
                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" />
                      <span className="text-sm font-medium">Severity:</span>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${getSeverityColor(result.severity)}`}>
                        {result.severity}
                      </span>
                    </div>
                    {result.model_used === "dummy" && (
                      <span className="text-xs px-2 py-1 rounded-full bg-warning/20 text-warning">
                        Demo mode — model not loaded
                      </span>
                    )}
                  </div>

                  {/* Treatment */}
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-success" />
                      Recommended Treatment
                    </h4>
                    <ul className="space-y-2">
                      {(result.treatment.length ? result.treatment : ["No specific treatment available."]).map((item, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm">
                          <span className="h-1.5 w-1.5 rounded-full bg-success mt-2 shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Prevention */}
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <Info className="h-4 w-4 text-info" />
                      Prevention Tips
                    </h4>
                    <ul className="space-y-2">
                      {(result.prevention.length ? result.prevention : ["Monitor crop condition and repeat analysis with clearer images if needed."]).map((item, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <span className="h-1.5 w-1.5 rounded-full bg-info mt-2 shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Diagnosis;
