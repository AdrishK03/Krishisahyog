import { Link } from "react-router-dom";
import { Sprout, Mail, Phone, MapPin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-primary text-primary-foreground">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-primary-foreground/20">
                <Sprout className="h-5 w-5" />
              </div>
              <span className="font-bold text-xl">KrishiSahyog</span>
            </div>
            <p className="text-primary-foreground/80 text-sm">
              Empowering farmers with AI-powered insights and real-time agricultural advisory.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2">
              {["Home", "Features", "Dashboard", "About Us"].map((item) => (
                <li key={item}>
                  <Link
                    to={item === "Home" ? "/" : `/${item.toLowerCase().replace(" ", "-")}`}
                    className="text-primary-foreground/80 hover:text-primary-foreground transition-colors text-sm"
                  >
                    {item}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Features */}
          <div>
            <h4 className="font-semibold mb-4">Features</h4>
            <ul className="space-y-2">
              {["Plant Diagnosis", "Soil Analysis", "Weather Insights", "Smart Assistant"].map((item) => (
                <li key={item}>
                  <span className="text-primary-foreground/80 text-sm">{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="font-semibold mb-4">Contact Us</h4>
            <ul className="space-y-3">
              <li className="flex items-center gap-2 text-sm text-primary-foreground/80">
                <Mail className="h-4 w-4" />
                support@krishisahyog.com
              </li>
              <li className="flex items-center gap-2 text-sm text-primary-foreground/80">
                <Phone className="h-4 w-4" />
                +91 1800-XXX-XXXX
              </li>
              <li className="flex items-center gap-2 text-sm text-primary-foreground/80">
                <MapPin className="h-4 w-4" />
                New Delhi, India
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-primary-foreground/20 mt-8 pt-8 text-center text-sm text-primary-foreground/60">
          <p>© 2024 KrishiSahyog. All rights reserved. Made with ❤️ for Indian Farmers</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
