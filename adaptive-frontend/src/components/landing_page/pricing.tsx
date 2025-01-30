import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function Pricing() {
  const plans = [
    {
      name: "Starter",
      price: "$49.99",
      description: "For small teams just starting with LLMs",
      features: ["1 user", "10 projects", "10GB storage", "Basic LLM support"],
    },
    {
      name: "Growth",
      price: "$149.99",
      description: "Ideal for growing teams utilizing LLM-driven workflows",
      features: [
        "5 users",
        "50 projects",
        "50GB storage",
        "Priority LLM support",
      ],
    },
    {
      name: "Enterprise",
      price: "Custom",
      description:
        "For organizations with large-scale LLM-driven infrastructure needs",
      features: [
        "Unlimited users",
        "Unlimited projects",
        "Unlimited storage",
        "24/7 dedicated LLM support",
      ],
    },
  ];

  return (
    <section
      id="pricing"
      className="w-full py-12 md:py-24 lg:py-32 bg-background"
    >
      <div className="container px-4 md:px-6">
        <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12 text-primary-600 dark:text-primary-200">
          Adaptive Pricing Plans
        </h2>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {plans.map((plan, index) => (
            <Card
              key={index}
              className="flex flex-col justify-between border border-border dark:border-border/50 bg-card"
            >
              <CardHeader>
                <CardTitle className="text-primary-600 dark:text-primary-200">
                  {plan.name}
                </CardTitle>
                <CardDescription className="text-muted-foreground">
                  {plan.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-primary-600 dark:text-primary-200">
                  {plan.price}
                </p>
                <p className="text-sm text-muted-foreground dark:text-muted-foreground">
                  per month
                </p>
                <ul className="mt-4 space-y-2">
                  {plan.features.map((feature, idx) => (
                    <li
                      key={idx}
                      className="flex items-center text-muted-foreground dark:text-muted-foreground"
                    >
                      <svg
                        className="w-4 h-4 mr-2 text-primary-600 dark:text-primary-200"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M5 13l4 4L19 7"
                        ></path>
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button className="w-full bg-primary-600 text-primary-50 hover:bg-primary-700 dark:bg-primary-500 dark:text-primary-50 dark:hover:bg-primary-600">
                  Choose Plan
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
