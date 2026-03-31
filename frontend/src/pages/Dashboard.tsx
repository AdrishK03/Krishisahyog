import { useEffect, useState } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import StatCard from "@/components/features/StatCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Leaf,
  Droplets,
  Thermometer,
  Cloud,
  TrendingUp,
  AlertTriangle,
  Sun,
  Wind,
  RefreshCw,
  CloudRain,
  CloudSnow,
  CloudLightning,
  Loader2,
  MapPin,
} from "lucide-react";

function wmoToCondition(code: number): { condition: string; Icon: React.ElementType } {
  if (code === 0) return { condition: "Clear Sky", Icon: Sun };
  if (code <= 2) return { condition: "Partly Cloudy", Icon: Sun };
  if (code === 3) return { condition: "Overcast", Icon: Cloud };
  if (code <= 49) return { condition: "Foggy", Icon: Cloud };
  if (code <= 55) return { condition: "Drizzle", Icon: CloudRain };
  if (code <= 65) return { condition: "Rainy", Icon: CloudRain };
  if (code <= 77) return { condition: "Snowy", Icon: CloudSnow };
  if (code <= 82) return { condition: "Rain Showers", Icon: CloudRain };
  if (code <= 86) return { condition: "Snow Showers", Icon: CloudSnow };
  if (code <= 99) return { condition: "Thunderstorm", Icon: CloudLightning };
  return { condition: "Unknown", Icon: Cloud };
}

interface DashboardWeather {
  location: string;
  temp: string;
  humidity: string;
  wind: string;
  condition: string;
  Icon: React.ElementType;
}

const soilData = [
  { label: "pH Level", value: "6.8", status: "optimal", icon: Droplets },
  { label: "Moisture", value: "42%", status: "good", icon: Droplets },
  { label: "Temperature", value: "24°C", status: "optimal", icon: Thermometer },
  { label: "Nitrogen", value: "45 ppm", status: "low", icon: Leaf },
];

const recentAlerts = [
  { type: "warning", message: "Low nitrogen levels detected in Field 2", time: "2 hours ago" },
  { type: "info", message: "Rainfall expected tomorrow - plan irrigation", time: "4 hours ago" },
  { type: "success", message: "Crop health analysis completed for wheat field", time: "1 day ago" },
];

const Dashboard = () => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [weatherLoading, setWeatherLoading] = useState(true);
  const [weatherData, setWeatherData] = useState<DashboardWeather>({
    location: "Detecting location...",
    temp: "--°C",
    humidity: "--%",
    wind: "-- km/h",
    condition: "Loading...",
    Icon: Cloud,
  });

  const fetchWeather = async () => {
    setWeatherLoading(true);
    if (!navigator.geolocation) {
      setWeatherData((prev) => ({
        ...prev,
        location: "Location unavailable",
        condition: "Geolocation not supported",
      }));
      setWeatherLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async ({ coords }) => {
        const { latitude, longitude } = coords;
        try {
          const geoRes = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`
          );
          const geoData = await geoRes.json();
          const city =
            geoData.address?.city ||
            geoData.address?.town ||
            geoData.address?.village ||
            geoData.address?.county ||
            "Your Location";

          const weatherRes = await fetch(
            `https://api.open-meteo.com/v1/forecast` +
              `?latitude=${latitude}&longitude=${longitude}` +
              `&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code` +
              `&timezone=auto`
          );
          const weatherJson = await weatherRes.json();
          const current = weatherJson.current;
          const { condition, Icon } = wmoToCondition(current.weather_code);

          setWeatherData({
            location: city,
            temp: `${Math.round(current.temperature_2m)}°C`,
            humidity: `${current.relative_humidity_2m}%`,
            wind: `${Math.round(current.wind_speed_10m)} km/h`,
            condition,
            Icon,
          });
        } catch {
          setWeatherData((prev) => ({
            ...prev,
            location: "Weather unavailable",
            condition: "Failed to load weather",
          }));
        } finally {
          setWeatherLoading(false);
        }
      },
      () => {
        setWeatherData((prev) => ({
          ...prev,
          location: "Location permission denied",
          condition: "Enable location to see live weather",
        }));
        setWeatherLoading(false);
      }
    );
  };

  useEffect(() => {
    fetchWeather();
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchWeather();
    setIsRefreshing(false);
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold">Farm Dashboard</h1>
            <p className="text-muted-foreground">Monitor your farm's health and get real-time insights</p>
          </div>
          <Button variant="outline" size="sm" className="w-fit" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            Refresh Data
          </Button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            icon={Leaf}
            label="Crop Health Score"
            value="87%"
            trend="+5%"
            trendUp
            color="success"
          />
          <StatCard
            icon={Droplets}
            label="Soil Moisture"
            value="42%"
            trend="-3%"
            trendUp={false}
            color="info"
          />
          <StatCard
            icon={TrendingUp}
            label="Yield Estimate"
            value="4.2 tons"
            trend="+12%"
            trendUp
            color="primary"
          />
          <StatCard
            icon={AlertTriangle}
            label="Active Alerts"
            value="2"
            color="warning"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Weather Card */}
          <Card className="lg:col-span-1">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Cloud className="h-5 w-5 text-info" />
                Weather Today
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground mb-4 flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5" />
                {weatherData.location}
              </p>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <p className="text-4xl font-bold">{weatherData.temp}</p>
                  <p className="text-muted-foreground">{weatherData.condition}</p>
                </div>
                {weatherLoading ? (
                  <Loader2 className="h-10 w-10 text-primary animate-spin" />
                ) : (
                  <weatherData.Icon className="h-16 w-16 text-accent" />
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-2 text-sm">
                  <Droplets className="h-4 w-4 text-info" />
                  <span className="text-muted-foreground">Humidity:</span>
                  <span className="font-medium">{weatherData.humidity}</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <Wind className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">Wind:</span>
                  <span className="font-medium">{weatherData.wind}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Soil Analysis Card */}
          <Card className="lg:col-span-2">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Thermometer className="h-5 w-5 text-primary" />
                Soil Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {soilData.map((item) => (
                  <div
                    key={item.label}
                    className={`p-4 rounded-xl border ${
                      item.status === "optimal"
                        ? "bg-success/5 border-success/20"
                        : item.status === "low"
                        ? "bg-warning/5 border-warning/20"
                        : "bg-secondary/50 border-border"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <item.icon className={`h-4 w-4 ${
                        item.status === "optimal" ? "text-success" : 
                        item.status === "low" ? "text-warning" : "text-muted-foreground"
                      }`} />
                      <span className="text-xs text-muted-foreground">{item.label}</span>
                    </div>
                    <p className="text-xl font-bold">{item.value}</p>
                    <span className={`text-xs capitalize ${
                      item.status === "optimal" ? "text-success" : 
                      item.status === "low" ? "text-warning" : "text-muted-foreground"
                    }`}>
                      {item.status}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Recent Alerts */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <AlertTriangle className="h-5 w-5 text-warning" />
              Recent Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentAlerts.map((alert, index) => (
                <div
                  key={index}
                  className={`flex items-start gap-3 p-3 rounded-lg ${
                    alert.type === "warning"
                      ? "bg-warning/10"
                      : alert.type === "success"
                      ? "bg-success/10"
                      : "bg-info/10"
                  }`}
                >
                  <div className={`h-2 w-2 rounded-full mt-2 ${
                    alert.type === "warning"
                      ? "bg-warning"
                      : alert.type === "success"
                      ? "bg-success"
                      : "bg-info"
                  }`} />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{alert.message}</p>
                    <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default Dashboard;
