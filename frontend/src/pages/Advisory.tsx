import { useState } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Wheat,
  Leaf,
  Droplets,
  Bug,
  Sun,
  Wind,
  Thermometer,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Calendar,
  RefreshCw,
  ArrowRight,
} from "lucide-react";

// Mock data - will be replaced with Flask API calls
const crops = [
  { id: "wheat", name: "Wheat", icon: Wheat },
  { id: "rice", name: "Rice", icon: Leaf },
  { id: "potato", name: "Potato", icon: Leaf },
  { id: "tomato", name: "Tomato", icon: Leaf },
];

const mockSensorData = {
  soilMoisture: 65,
  soilPH: 6.8,
  temperature: 28,
  humidity: 72,
  nitrogen: 45,
  phosphorus: 38,
  potassium: 52,
};

const mockWeatherAlerts = [
  { type: "warning", message: "Heavy rainfall expected in next 48 hours", icon: AlertTriangle },
  { type: "info", message: "Temperature dropping to 18°C tonight", icon: Thermometer },
];

const mockPestAlerts = [
  { pest: "Aphids", risk: "high", crop: "wheat" },
  { pest: "Stem Borer", risk: "medium", crop: "rice" },
];

const Advisory = () => {
  const [selectedCrop, setSelectedCrop] = useState<string>("");
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleGetPrediction = async () => {
    if (!selectedCrop) return;
    
    setIsLoading(true);
    
    // TODO: Replace with actual Flask API call
    // const response = await axios.post(`${API_URL}/predict`, {
    //   crop: selectedCrop,
    //   sensorData: mockSensorData,
    // });
    
    // Simulated prediction response
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const mockPredictions: Record<string, any> = {
      wheat: {
        yield: "3.2 tons/hectare",
        confidence: 87,
        health: "Good",
        advisory: [
          "Apply 50kg urea per hectare in the next week",
          "Irrigation recommended every 5 days",
          "Monitor for rust disease symptoms",
        ],
        fertilizer: { nitrogen: "+20%", phosphorus: "Optimal", potassium: "-10%" },
        irrigation: "Next irrigation in 3 days",
      },
      rice: {
        yield: "4.5 tons/hectare",
        confidence: 82,
        health: "Moderate",
        advisory: [
          "Increase water level to 5cm",
          "Apply potash fertilizer",
          "Watch for brown plant hopper",
        ],
        fertilizer: { nitrogen: "Optimal", phosphorus: "-15%", potassium: "+25%" },
        irrigation: "Maintain continuous flooding",
      },
      potato: {
        yield: "25 tons/hectare",
        confidence: 79,
        health: "Good",
        advisory: [
          "Hill up soil around plants",
          "Apply fungicide for late blight prevention",
          "Reduce irrigation frequency",
        ],
        fertilizer: { nitrogen: "-10%", phosphorus: "+15%", potassium: "Optimal" },
        irrigation: "Next irrigation in 5 days",
      },
      tomato: {
        yield: "35 tons/hectare",
        confidence: 84,
        health: "Good",
        advisory: [
          "Stake and prune suckers",
          "Apply calcium to prevent blossom end rot",
          "Monitor for tomato hornworm",
        ],
        fertilizer: { nitrogen: "Optimal", phosphorus: "Optimal", potassium: "+10%" },
        irrigation: "Drip irrigation recommended daily",
      },
    };
    
    setPrediction(mockPredictions[selectedCrop]);
    setIsLoading(false);
  };

  const getStatusColor = (value: string) => {
    if (value.includes("+")) return "text-amber-600";
    if (value.includes("-")) return "text-red-500";
    return "text-primary";
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold">Crop Advisory</h1>
            <p className="text-muted-foreground mt-1">
              AI-powered recommendations based on your field conditions
            </p>
          </div>
          <Badge variant="outline" className="gap-2 w-fit">
            <Calendar className="h-4 w-4" />
            Last updated: Today
          </Badge>
        </div>

        {/* Crop Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wheat className="h-5 w-5 text-primary" />
              Select Your Crop
            </CardTitle>
            <CardDescription>
              Choose a crop to get personalized ML-powered predictions and recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col sm:flex-row gap-4">
              <Select value={selectedCrop} onValueChange={setSelectedCrop}>
                <SelectTrigger className="w-full sm:w-64">
                  <SelectValue placeholder="Select crop type" />
                </SelectTrigger>
                <SelectContent>
                  {crops.map((crop) => (
                    <SelectItem key={crop.id} value={crop.id}>
                      <div className="flex items-center gap-2">
                        <crop.icon className="h-4 w-4" />
                        {crop.name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={handleGetPrediction}
                disabled={!selectedCrop || isLoading}
                className="gap-2"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Get Prediction
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Sensor Data Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10">
                <Droplets className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Soil Moisture</p>
                <p className="text-xl font-bold">{mockSensorData.soilMoisture}%</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <Leaf className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Soil pH</p>
                <p className="text-xl font-bold">{mockSensorData.soilPH}</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-orange-500/10">
                <Thermometer className="h-5 w-5 text-orange-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Temperature</p>
                <p className="text-xl font-bold">{mockSensorData.temperature}°C</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10">
                <Wind className="h-5 w-5 text-cyan-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Humidity</p>
                <p className="text-xl font-bold">{mockSensorData.humidity}%</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Prediction Results */}
        {prediction && (
          <div className="grid md:grid-cols-2 gap-6 animate-fade-in">
            {/* Yield Prediction */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  Yield Prediction
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Expected Yield</span>
                  <span className="text-2xl font-bold text-primary">{prediction.yield}</span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Confidence</span>
                    <span>{prediction.confidence}%</span>
                  </div>
                  <Progress value={prediction.confidence} className="h-2" />
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={prediction.health === "Good" ? "default" : "secondary"}>
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Crop Health: {prediction.health}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Fertilizer Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Leaf className="h-5 w-5 text-primary" />
                  Fertilizer Adjustments
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(prediction.fertilizer).map(([nutrient, value]) => (
                    <div key={nutrient} className="flex items-center justify-between">
                      <span className="capitalize text-muted-foreground">{nutrient}</span>
                      <span className={`font-medium ${getStatusColor(value as string)}`}>
                        {value as string}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 p-3 rounded-lg bg-primary/5 border border-primary/20">
                  <p className="text-sm flex items-center gap-2">
                    <Droplets className="h-4 w-4 text-primary" />
                    {prediction.irrigation}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Advisory Actions */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-primary" />
                  Recommended Actions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid sm:grid-cols-3 gap-4">
                  {prediction.advisory.map((advice: string, index: number) => (
                    <div
                      key={index}
                      className="p-4 rounded-lg bg-secondary/50 border border-border hover:border-primary/50 transition-colors"
                    >
                      <div className="flex items-start gap-3">
                        <div className="h-6 w-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium shrink-0">
                          {index + 1}
                        </div>
                        <p className="text-sm">{advice}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Alerts Section */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Weather Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sun className="h-5 w-5 text-amber-500" />
                Weather Alerts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {mockWeatherAlerts.map((alert, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg flex items-start gap-3 ${
                    alert.type === "warning" ? "bg-amber-500/10 border border-amber-500/30" : "bg-blue-500/10 border border-blue-500/30"
                  }`}
                >
                  <alert.icon className={`h-5 w-5 shrink-0 ${
                    alert.type === "warning" ? "text-amber-500" : "text-blue-500"
                  }`} />
                  <p className="text-sm">{alert.message}</p>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Pest/Disease Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bug className="h-5 w-5 text-red-500" />
                Pest & Disease Risk
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {mockPestAlerts.map((alert, index) => (
                <div
                  key={index}
                  className="p-3 rounded-lg bg-secondary/50 border border-border flex items-center justify-between"
                >
                  <div>
                    <p className="font-medium">{alert.pest}</p>
                    <p className="text-sm text-muted-foreground">Affects: {alert.crop}</p>
                  </div>
                  <Badge variant={alert.risk === "high" ? "destructive" : "secondary"}>
                    {alert.risk} risk
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Advisory;
