import { Cpu, Lock, Sparkles, Zap } from "lucide-react";

export default function FeaturesSection() {
  const features = [
    {
      icon: <Zap className="size-4" />,
      title: "Lightning Fast",
      description: "Experience unparalleled speed with our optimized platform.",
    },
    {
      icon: <Cpu className="size-4" />,
      title: "Intuitive Interface",
      description: "Navigate with ease through our user-friendly design.",
    },
    {
      icon: <Lock className="size-4" />,
      title: "Security",
      description: "Work seamlessly with your team in real-time.",
    },
    {
      icon: <Sparkles className="size-4" />,
      title: "AI Powered",
      description: "Experience the power of AI to enhance your workflow.",
    },
  ];

  return (
    <section className="overflow-hidden py-16 md:py-32">
      <div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-12">
        <div className="relative z-10 max-w-2xl">
          <h2 className="text-4xl font-semibold lg:text-5xl">
            Built for Scaling teams
          </h2>
          <p className="mt-6 text-lg">
            Empower your team with workflows that adapt to your needs, whether
            you prefer git synchronization or a AI Agents interface.
          </p>
        </div>

        <div className="relative -mx-4 rounded-3xl p-3 md:-mx-12 lg:col-span-3">
          <div className="perspective-midrange">
            <div className="rotate-x-6 -skew-2">
              <div className="aspect-88/36 relative">
                <div className="bg-radial-[at_75%_25%] to-background z-1 -inset-17 absolute from-transparent to-75%"></div>

                <img
                  src="/adaptive-chat.jpeg"
                  className="hidden dark:block"
                  alt="payments illustration dark"
                />
                <img
                  src="/adaptive-chat-light.jpeg"
                  className="dark:hidden"
                  alt="payments illustration light"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="relative mx-auto grid grid-cols-2 gap-x-3 gap-y-6 sm:gap-8 lg:grid-cols-4">
          {features.map((feature, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-center gap-2">
                {feature.icon}
                <h3 className="text-sm font-medium">{feature.title}</h3>
              </div>
              <p className="text-muted-foreground text-sm">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
