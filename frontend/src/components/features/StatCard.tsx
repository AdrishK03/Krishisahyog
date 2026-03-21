import type { LucideIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
  trend?: string;
  trendUp?: boolean;
  color?: "primary" | "success" | "warning" | "info" | "accent";
}

const colorVariants = {
  primary: "bg-primary/10 text-primary",
  success: "bg-success/10 text-success",
  warning: "bg-warning/10 text-warning",
  info: "bg-info/10 text-info",
  accent: "bg-accent/10 text-accent-foreground",
};

const StatCard = ({ icon: Icon, label, value, trend, trendUp, color = "primary" }: StatCardProps) => {
  return (
    <Card className="border-border/50 hover:shadow-md transition-all duration-300">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className={cn("p-2.5 rounded-lg", colorVariants[color])}>
            <Icon className="h-5 w-5" />
          </div>
          {trend && (
            <span className={cn(
              "text-xs font-medium px-2 py-1 rounded-full",
              trendUp ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"
            )}>
              {trend}
            </span>
          )}
        </div>
        <div className="mt-4">
          <p className="text-2xl font-bold text-foreground">{value}</p>
          <p className="text-sm text-muted-foreground mt-1">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default StatCard;
