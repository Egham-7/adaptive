import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Cpu, Zap, Search } from "lucide-react";

export default function Features() {
  const features = [
    {
      icon: <Cpu className="h-6 w-6" />,
      title: "Efficient Model Selection",
      description:
        "Automatically route queries to the most efficient AI models, saving you on computing costs without compromising on quality.",
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: "Fast Inference",
      description:
        "Experience reduced latency with optimized GPU scheduling, delivering faster responses for your AI applications.",
    },
    {
      icon: <Search className="h-6 w-6" />,
      title: "Real-Time Benchmarking",
      description:
        "Track performance metrics and cost savings in real time, ensuring you get the best ROI with every query processed.",
    },
  ];

  return (
    <section
      id="features"
      className="w-full py-12 md:py-24 lg:py-32 bg-primary-50 dark:bg-primary-950"
    >
      <div className="container px-4 md:px-6">
        <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12 text-primary-600 dark:text-primary-200">
          Key Features
        </h2>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-card text-card-foreground dark:bg-card dark:text-card-foreground"
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-primary-600 dark:text-primary-200">
                  {feature.icon}
                  <span>{feature.title}</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-primary-500 dark:text-primary-400">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
