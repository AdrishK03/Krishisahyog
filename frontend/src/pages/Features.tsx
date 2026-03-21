import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import FeatureCard from "@/components/features/FeatureCard";
import {
  Leaf,
  Mountain,
  Cloud,
  MessageCircle,
  ShoppingCart,
  Wheat,
  Camera,
  BarChart3,
  Globe,
  Smartphone,
} from "lucide-react";

const allFeatures = [
  {
    icon: Leaf,
    title: "Plant Health Diagnosis",
    description: "Upload plant images for AI-powered disease and pest detection using advanced computer vision technology.",
  },
  {
    icon: Mountain,
    title: "Soil Monitoring",
    description: "Real-time soil pH, moisture, and temperature monitoring through smart IoT sensors integrated with your farm.",
  },
  {
    icon: Cloud,
    title: "Weather Insights",
    description: "Get localized weather forecasts and agricultural alerts with 7-day predictions for better planning.",
  },
  {
    icon: MessageCircle,
    title: "Smart Assistant",
    description: "AI-powered chatbot providing personalized farming advice in multiple languages including Hindi.",
  },
  {
    icon: ShoppingCart,
    title: "Market Prices",
    description: "Real-time crop prices from APMC markets across India with price trend analysis and selling recommendations.",
  },
  {
    icon: Wheat,
    title: "Crop Advisory",
    description: "Get personalized crop recommendations based on your soil conditions, weather patterns, and market demand.",
  },
  {
    icon: Camera,
    title: "Image Analysis",
    description: "Advanced image recognition to identify crop diseases, pests, and nutrient deficiencies with 95%+ accuracy.",
  },
  {
    icon: BarChart3,
    title: "Yield Prediction",
    description: "Machine learning models to predict crop yields based on historical data, weather, and soil conditions.",
  },
  {
    icon: Globe,
    title: "Multi-language Support",
    description: "Full support for Hindi, English, and regional languages making the platform accessible to all farmers.",
  },
  {
    icon: Smartphone,
    title: "Mobile Optimized",
    description: "Fully responsive design that works perfectly on smartphones, tablets, and desktop computers.",
  },
];

const Features = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="py-20 bg-gradient-to-br from-secondary/50 via-background to-primary/5">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Explore Our <span className="text-primary">Smart Features</span>
          </h1>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Comprehensive tools and AI-powered insights to revolutionize your farming practices and maximize productivity.
          </p>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 bg-background">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {allFeatures.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                delay={index * 50}
              />
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Features;
