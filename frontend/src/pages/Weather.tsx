import { useEffect, useState } from "react";
import DashboardLayout from "@/components/layout/DashboardLayout";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Cloud,
  Sun,
  CloudRain,
  Wind,
  Droplets,
  Thermometer,
  Eye,
  AlertTriangle,
  CloudSnow,
  CloudLightning,
  Loader2,
  MapPin,
} from "lucide-react";

// ─── WMO weather code → { condition, Icon } ───────────────────────────────
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

const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

// ─── Types ─────────────────────────────────────────────────────────────────
interface CurrentWeather {
  temp: number;
  feelsLike: number;
  humidity: number;
  windSpeed: number;
  visibility: number;
  uvIndex: number;
  condition: string;
  Icon: React.ElementType;
}

interface ForecastDay {
  day: string;
  high: number;
  low: number;
  condition: string;
  Icon: React.ElementType;
  rain: number;
}

// ─── Component ─────────────────────────────────────────────────────────────
const Weather = () => {
  const [location, setLocation] = useState<string>("Detecting location…");
  const [current, setCurrent] = useState<CurrentWeather | null>(null);
  const [forecast, setForecast] = useState<ForecastDay[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser.");
      setLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async ({ coords }) => {
        const { latitude, longitude } = coords;

        try {
          // 1️⃣  Reverse-geocode to get city name (Open-Meteo geocoding)
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
          setLocation(city);

          // 2️⃣  Fetch weather from Open-Meteo (NO API KEY REQUIRED)
          const weatherRes = await fetch(
            `https://api.open-meteo.com/v1/forecast` +
              `?latitude=${latitude}&longitude=${longitude}` +
              `&current=temperature_2m,apparent_temperature,relative_humidity_2m,` +
              `wind_speed_10m,weather_code,visibility` +
              `&daily=weather_code,temperature_2m_max,temperature_2m_min,` +
              `precipitation_probability_max,uv_index_max` +
              `&timezone=auto&forecast_days=5`
          );
          const w = await weatherRes.json();

          const c = w.current;
          const { condition, Icon } = wmoToCondition(c.weather_code);

          setCurrent({
            temp: Math.round(c.temperature_2m),
            feelsLike: Math.round(c.apparent_temperature),
            humidity: c.relative_humidity_2m,
            windSpeed: Math.round(c.wind_speed_10m),
            // Open-Meteo returns visibility in metres; convert to km
            visibility: c.visibility != null ? Math.round(c.visibility / 1000) : 10,
            uvIndex: Math.round(w.daily.uv_index_max[0] ?? 0),
            condition,
            Icon,
          });

          const days: ForecastDay[] = w.daily.time.map(
            (dateStr: string, i: number) => {
              const dayName =
                i === 0
                  ? "Today"
                  : i === 1
                  ? "Tomorrow"
                  : DAYS[new Date(dateStr).getDay()];
              const { condition: dc, Icon: DIcon } = wmoToCondition(
                w.daily.weather_code[i]
              );
              return {
                day: dayName,
                high: Math.round(w.daily.temperature_2m_max[i]),
                low: Math.round(w.daily.temperature_2m_min[i]),
                condition: dc,
                Icon: DIcon,
                rain: w.daily.precipitation_probability_max[i] ?? 0,
              };
            }
          );
          setForecast(days);
        } catch (e) {
          setError("Failed to fetch weather data. Please try again.");
        } finally {
          setLoading(false);
        }
      },
      () => {
        setError(
          "Location access denied. Please allow location permission in your browser and reload."
        );
        setLoading(false);
      }
    );
  }, []);

  // ─── Loading state ────────────────────────────────────────────────────────
  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <Loader2 className="h-10 w-10 text-primary animate-spin" />
          <p className="text-muted-foreground">Detecting your location and fetching weather…</p>
        </div>
      </DashboardLayout>
    );
  }

  // ─── Error state ──────────────────────────────────────────────────────────
  if (error || !current) {
    return (
      <DashboardLayout>
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <AlertTriangle className="h-10 w-10 text-warning" />
          <p className="text-muted-foreground text-center max-w-sm">{error ?? "No data available."}</p>
        </div>
      </DashboardLayout>
    );
  }

  // ─── Agricultural alerts derived from forecast ────────────────────────────
  const heavyRainDay = forecast.find((d) => d.rain >= 70 && d.day !== "Today");
  const goodSowingDay = forecast.find(
    (d, i) => i >= 2 && d.rain < 30 && d.high >= 20
  );

  const alerts = [
    heavyRainDay && {
      type: "warning",
      title: "Heavy Rainfall Expected",
      description: `High chance of rainfall on ${heavyRainDay.day} (${heavyRainDay.rain}%). Secure crops and ensure proper drainage.`,
      time: `Valid: ${heavyRainDay.day}`,
    },
    goodSowingDay && {
      type: "info",
      title: "Ideal Sowing Conditions",
      description: `Low rain probability and warm temperatures on ${goodSowingDay.day}. Plan sowing activities accordingly.`,
      time: `Valid: ${goodSowingDay.day} onwards`,
    },
  ].filter(Boolean) as { type: string; title: string; description: string; time: string }[];

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <Cloud className="h-8 w-8 text-primary" />
            Weather Insights
          </h1>
          <p className="text-muted-foreground mt-1 flex items-center gap-1">
            <MapPin className="h-4 w-4" />
            {location} — localized forecasts and agricultural alerts
          </p>
        </div>

        {/* Current Weather */}
        <Card className="bg-gradient-to-br from-info/10 to-primary/5 border-info/20">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-6">
                <div className="h-24 w-24 rounded-full bg-accent/20 flex items-center justify-center">
                  <current.Icon className="h-12 w-12 text-accent" />
                </div>
                <div>
                  <p className="text-5xl font-bold">{current.temp}°C</p>
                  <p className="text-muted-foreground">{current.condition}</p>
                  <p className="text-sm text-muted-foreground">
                    Feels like {current.feelsLike}°C
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { icon: Droplets, label: "Humidity", value: `${current.humidity}%` },
                  { icon: Wind, label: "Wind", value: `${current.windSpeed} km/h` },
                  { icon: Eye, label: "Visibility", value: `${current.visibility} km` },
                  { icon: Thermometer, label: "UV Index", value: String(current.uvIndex) },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-2 text-sm">
                    <item.icon className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-muted-foreground">{item.label}</p>
                      <p className="font-medium">{item.value}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 5-Day Forecast */}
        <Card>
          <CardHeader>
            <CardTitle>5-Day Forecast</CardTitle>
            <CardDescription>Plan your farming activities with weather predictions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {forecast.map((day) => (
                <div
                  key={day.day}
                  className="p-4 rounded-xl bg-secondary/50 text-center hover:bg-secondary transition-colors"
                >
                  <p className="font-medium mb-2">{day.day}</p>
                  <day.Icon className="h-10 w-10 mx-auto text-primary mb-2" />
                  <p className="text-sm text-muted-foreground">{day.condition}</p>
                  <div className="mt-2">
                    <span className="font-bold">{day.high}°</span>
                    <span className="text-muted-foreground"> / {day.low}°</span>
                  </div>
                  <div className="flex items-center justify-center gap-1 mt-2 text-xs text-info">
                    <Droplets className="h-3 w-3" />
                    {day.rain}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Weather Alerts */}
        {alerts.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-warning" />
                Agricultural Alerts
              </CardTitle>
              <CardDescription>Important weather warnings for your farm</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.map((alert, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-xl border ${
                      alert.type === "warning"
                        ? "bg-warning/5 border-warning/20"
                        : "bg-info/5 border-info/20"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <AlertTriangle
                        className={`h-5 w-5 mt-0.5 ${
                          alert.type === "warning" ? "text-warning" : "text-info"
                        }`}
                      />
                      <div>
                        <h4 className="font-semibold">{alert.title}</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          {alert.description}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">{alert.time}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
};

export default Weather;