import DashboardLayout from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ShoppingCart, TrendingUp, ArrowUpRight, ArrowDownRight, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

const marketPrices = [
  { crop: "Wheat", price: 2250, unit: "per quintal", change: 3.5, trending: "up" },
  { crop: "Rice (Basmati)", price: 3800, unit: "per quintal", change: -1.2, trending: "down" },
  { crop: "Maize", price: 1950, unit: "per quintal", change: 2.1, trending: "up" },
  { crop: "Soybean", price: 4200, unit: "per quintal", change: 5.8, trending: "up" },
  { crop: "Cotton", price: 6800, unit: "per quintal", change: -2.3, trending: "down" },
  { crop: "Mustard", price: 5100, unit: "per quintal", change: 1.5, trending: "up" },
  { crop: "Groundnut", price: 5600, unit: "per quintal", change: 0.8, trending: "up" },
  { crop: "Sugarcane", price: 350, unit: "per quintal", change: 0, trending: "stable" },
];

const marketInsights = [
  {
    title: "Best Time to Sell Wheat",
    description: "Wheat prices are expected to rise by 5-8% in the next 2 weeks due to increased demand.",
    recommendation: "Hold",
  },
  {
    title: "Cotton Market Alert",
    description: "Cotton prices showing downward trend. Consider alternative storage or early selling.",
    recommendation: "Sell Soon",
  },
  {
    title: "Soybean Price Surge",
    description: "Export demand driving soybean prices up. Good time to negotiate contracts.",
    recommendation: "Sell Now",
  },
];

const Market = () => {
  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
              <ShoppingCart className="h-8 w-8 text-primary" />
              Market Prices
            </h1>
            <p className="text-muted-foreground mt-1">
              Real-time crop prices and market trends
            </p>
          </div>
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Update Prices
          </Button>
        </div>

        {/* Price Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {marketPrices.map((item) => (
            <Card key={item.crop} className="hover:shadow-md transition-all duration-300">
              <CardContent className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <h3 className="font-semibold">{item.crop}</h3>
                  {item.trending === "up" ? (
                    <div className="flex items-center text-success text-sm">
                      <ArrowUpRight className="h-4 w-4" />
                      {item.change}%
                    </div>
                  ) : item.trending === "down" ? (
                    <div className="flex items-center text-destructive text-sm">
                      <ArrowDownRight className="h-4 w-4" />
                      {Math.abs(item.change)}%
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">Stable</div>
                  )}
                </div>
                <p className="text-2xl font-bold">₹{item.price.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">{item.unit}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Market Insights */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Market Insights
            </CardTitle>
            <CardDescription>AI-powered market analysis and recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {marketInsights.map((insight, index) => (
                <div
                  key={index}
                  className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl bg-secondary/50 gap-4"
                >
                  <div>
                    <h4 className="font-semibold">{insight.title}</h4>
                    <p className="text-sm text-muted-foreground mt-1">{insight.description}</p>
                  </div>
                  <span className={`px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap ${
                    insight.recommendation === "Sell Now"
                      ? "bg-success/10 text-success"
                      : insight.recommendation === "Sell Soon"
                      ? "bg-warning/10 text-warning"
                      : "bg-info/10 text-info"
                  }`}>
                    {insight.recommendation}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Price Trend Note */}
        <Card className="bg-gradient-to-br from-accent/10 to-primary/5">
          <CardContent className="p-6">
            <div className="flex items-start gap-4">
              <div className="h-12 w-12 rounded-xl bg-accent/20 flex items-center justify-center shrink-0">
                <TrendingUp className="h-6 w-6 text-accent" />
              </div>
              <div>
                <h3 className="font-semibold mb-1">Market Update</h3>
                <p className="text-sm text-muted-foreground">
                  Prices are updated daily from APMC markets across India. For local mandi rates, 
                  please check with your nearest agricultural market. Historical trends and 
                  predictions are based on government data and market analysis.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default Market;
