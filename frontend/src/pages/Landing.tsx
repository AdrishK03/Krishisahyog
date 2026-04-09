import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import FeatureCard from "@/components/features/FeatureCard";
import heroImage from "@/assets/hero-farm.jpg";
import {
  Leaf,
  Mountain,
  Cloud,
  MessageCircle,
  ShoppingCart,
  Wheat,
  ArrowRight,
  CheckCircle,
  Users,
  TrendingUp,
  Shield,
} from "lucide-react";

const features = [
  {
    icon: Leaf,
    title: "Plant Health Diagnosis",
    description: "Upload plant images for AI-powered disease detection and get instant treatment recommendations.",
  },
  {
    icon: Mountain,
    title: "Soil Analysis",
    description: "Real-time soil monitoring with pH, moisture, and nutrient level tracking through smart sensors.",
  },
  {
    icon: Cloud,
    title: "Weather Insights",
    description: "Localized weather forecasts with agricultural alerts to plan your farming activities.",
  },
  {
    icon: MessageCircle,
    title: "Smart Assistant",
    description: "Get personalized farming advice through our AI-powered chatbot in your preferred language.",
  },
  {
    icon: ShoppingCart,
    title: "Market Prices",
    description: "Stay updated with real-time crop prices and market trends to maximize your profits.",
  },
  {
    icon: Wheat,
    title: "Crop Advisory",
    description: "Personalized crop recommendations based on your soil, weather, and farming conditions.",
  },
];

const stats = [
  { value: "15+", label: "States Covered" },
  { value: "90%+", label: "Accuracy Rate" },
];

const Landing = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Background Image with Overlay */}
        <div className="absolute inset-0 z-0">
          <img
            src={heroImage}
            alt="Lush green farming fields"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-primary/95 via-primary/80 to-primary/60" />
        </div>

        <div className="container mx-auto px-4 py-20 md:py-32 relative z-10">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-primary-foreground mb-6 animate-slide-up leading-tight">
              Smart Agricultural
              <span className="block text-accent">Advisory System</span>
            </h1>
            
            <p className="text-lg md:text-xl text-primary-foreground/80 mb-8 max-w-2xl animate-slide-up" style={{ animationDelay: "100ms" }}>
              Empowering farmers with AI-powered insights, real-time data, and personalized recommendations for better yields and sustainable farming.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 animate-slide-up" style={{ animationDelay: "200ms" }}>
              <Button variant="accent" size="xl" asChild className="group">
                <Link to="/register">
                  Get Started Free
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Link>
              </Button>
              <Button variant="outline" size="xl" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10 hover:text-primary-foreground" asChild>
                <Link to="/features">Explore Features</Link>
              </Button>
            </div>
          </div>
        </div>

        {/* Wave Decoration */}
        <div className="absolute bottom-0 left-0 right-0 z-10">
          <svg viewBox="0 0 1440 120" className="w-full fill-background">
            <path d="M0,64L80,69.3C160,75,320,85,480,80C640,75,800,53,960,48C1120,43,1280,53,1360,58.7L1440,64L1440,120L1360,120C1280,120,1120,120,960,120C800,120,640,120,480,120C320,120,160,120,80,120L0,120Z" />
          </svg>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 bg-background relative z-20 -mt-1">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {stats.map((stat, index) => (
              <div
                key={stat.label}
                className="text-center p-6 rounded-xl bg-card border border-border/50 shadow-sm animate-scale-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <p className="text-3xl md:text-4xl font-bold text-primary mb-2">{stat.value}</p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-background">
        <div className="container mx-auto px-4">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Everything You Need for
              <span className="text-primary"> Smart Farming</span>
            </h2>
            <p className="text-muted-foreground text-lg">
              Comprehensive tools and insights to help you make informed decisions and maximize your agricultural success.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                delay={index * 100}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 bg-secondary/30">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                Why Choose <span className="text-primary">KrishiSahyog</span>?
              </h2>
              <div className="space-y-6">
                {[
                  { icon: Users, title: "Farmer-Friendly Design", desc: "Simple interface with support for multiple languages including Hindi" },
                  { icon: TrendingUp, title: "Increase Your Yields", desc: "Data-driven recommendations to improve crop productivity by up to 30%" },
                  { icon: Shield, title: "Trusted & Secure", desc: "Your farm data is protected with enterprise-grade security" },
                ].map((item) => (
                  <div key={item.title} className="flex gap-4">
                    <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                      <item.icon className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg mb-1">{item.title}</h3>
                      <p className="text-muted-foreground">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative">
              <div className="aspect-square rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 p-8 flex items-center justify-center">
                <div className="text-center">
                  <Wheat className="h-24 w-24 text-primary mx-auto mb-4 animate-float" />
                  <p className="text-2xl font-bold text-foreground">Start Growing</p>
                  <p className="text-muted-foreground">Better Today</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-hero text-primary-foreground">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to Transform Your Farming?
          </h2>
          <p className="text-lg text-primary-foreground/80 mb-8 max-w-2xl mx-auto">
            Join thousands of farmers who are already using KrishiSahyog to improve their yields and make smarter farming decisions.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="accent" size="xl" asChild>
              <Link to="/register">
                Start Free Trial
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            <Button variant="outline" size="xl" className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10 hover:text-primary-foreground" asChild>
              <Link to="/login">Sign In</Link>
            </Button>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;
