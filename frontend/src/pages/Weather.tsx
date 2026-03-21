import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Cloud, Sun, CloudRain, Wind, Droplets, Thermometer, Eye, AlertTriangle } from "lucide-react";

const currentWeather = {
  temp: 28,
  feelsLike: 31,
  humidity: 65,
  windSpeed: 12,
  visibility: 10,
  uvIndex: 6,
  condition: "Partly Cloudy",
  icon: Sun,
};

const forecast = [
  { day: "Today", high: 28, low: 22, condition: "Partly Cloudy", icon: Sun, rain: 10 },
  { day: "Tomorrow", high: 26, low: 20, condition: "Rainy", icon: CloudRain, rain: 80 },
  { day: "Wednesday", high: 24, low: 19, condition: "Rainy", icon: CloudRain, rain: 70 },
  { day: "Thursday", high: 27, low: 21, condition: "Cloudy", icon: Cloud, rain: 30 },
  { day: "Friday", high: 29, low: 23, condition: "Sunny", icon: Sun, rain: 5 },
];

const alerts = [
  {
    type: "warning",
    title: "Heavy Rainfall Expected",
    description: "70-80mm rainfall expected tomorrow. Secure crops and ensure proper drainage.",
    time: "Valid: Tomorrow 6 AM - 6 PM",
  },
  {
    type: "info",
    title: "Ideal Sowing Conditions",
    description: "Post-rain conditions will be ideal for wheat sowing. Plan accordingly.",
    time: "Valid: Thursday onwards",
  },
];

const Weather = () => {
  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
            <Cloud className="h-8 w-8 text-primary" />
            Weather Insights
          </h1>
          <p className="text-muted-foreground mt-1">
            Localized weather forecasts and agricultural alerts
          </p>
        </div>

        {/* Current Weather */}
        <Card className="bg-gradient-to-br from-info/10 to-primary/5 border-info/20">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-6">
                <div className="h-24 w-24 rounded-full bg-accent/20 flex items-center justify-center">
                  <currentWeather.icon className="h-12 w-12 text-accent" />
                </div>
                <div>
                  <p className="text-5xl font-bold">{currentWeather.temp}°C</p>
                  <p className="text-muted-foreground">{currentWeather.condition}</p>
                  <p className="text-sm text-muted-foreground">Feels like {currentWeather.feelsLike}°C</p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { icon: Droplets, label: "Humidity", value: `${currentWeather.humidity}%` },
                  { icon: Wind, label: "Wind", value: `${currentWeather.windSpeed} km/h` },
                  { icon: Eye, label: "Visibility", value: `${currentWeather.visibility} km` },
                  { icon: Thermometer, label: "UV Index", value: currentWeather.uvIndex.toString() },
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
                  <day.icon className="h-10 w-10 mx-auto text-primary mb-2" />
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
                    <AlertTriangle className={`h-5 w-5 mt-0.5 ${
                      alert.type === "warning" ? "text-warning" : "text-info"
                    }`} />
                    <div>
                      <h4 className="font-semibold">{alert.title}</h4>
                      <p className="text-sm text-muted-foreground mt-1">{alert.description}</p>
                      <p className="text-xs text-muted-foreground mt-2">{alert.time}</p>
                    </div>
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

export default Weather;
