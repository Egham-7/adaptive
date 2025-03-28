/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
  	extend: {
  		colors: {
  			primary: {
  				'50': 'hsl(222 89% 97%)',
  				'100': 'hsl(222 89% 94%)',
  				'200': 'hsl(222 89% 86%)',
  				'300': 'hsl(222 89% 78%)',
  				'400': 'hsl(222 89% 68%)',
  				'500': 'hsl(222 89% 51%)',
  				'600': 'hsl(222 89% 45%)',
  				'700': 'hsl(222 89% 38%)',
  				'800': 'hsl(222 89% 32%)',
  				'900': 'hsl(222 89% 25%)',
  				'950': 'hsl(222 89% 16%)',
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				'50': 'hsl(217 32% 97%)',
  				'100': 'hsl(217 32% 94%)',
  				'200': 'hsl(217 32% 86%)',
  				'300': 'hsl(217 32% 78%)',
  				'400': 'hsl(217 32% 68%)',
  				'500': 'hsl(217 32% 51%)',
  				'600': 'hsl(217 32% 45%)',
  				'700': 'hsl(217 32% 38%)',
  				'800': 'hsl(217 32% 32%)',
  				'900': 'hsl(217 32% 25%)',
  				'950': 'hsl(217 32% 16%)',
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			accent: {
  				'50': 'hsl(217 32% 97%)',
  				'100': 'hsl(217 32% 94%)',
  				'200': 'hsl(217 32% 86%)',
  				'300': 'hsl(217 32% 78%)',
  				'400': 'hsl(217 32% 68%)',
  				'500': 'hsl(217 32% 51%)',
  				'600': 'hsl(217 32% 45%)',
  				'700': 'hsl(217 32% 38%)',
  				'800': 'hsl(217 32% 32%)',
  				'900': 'hsl(217 32% 25%)',
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			},
  			sidebar: {
  				DEFAULT: 'hsl(var(--sidebar-background))',
  				foreground: 'hsl(var(--sidebar-foreground))',
  				primary: 'hsl(var(--sidebar-primary))',
  				'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
  				accent: 'hsl(var(--sidebar-accent))',
  				'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
  				border: 'hsl(var(--sidebar-border))',
  				ring: 'hsl(var(--sidebar-ring))'
  			}
  		},
  		fontFamily: {
  			sans: [
  				'Plus Jakarta Sans',
  				'Inter',
  				'system-ui',
  				'sans-serif'
  			],
  			display: [
  				'Cabinet Grotesk',
  				'Lexend',
  				'sans-serif'
  			],
  			mono: [
  				'JetBrains Mono',
  				'Menlo',
  				'monospace'
  			]
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 0.25rem)',
  			sm: 'calc(var(--radius) - 0.5rem)'
  		},
  		boxShadow: {
  			subtle: '0 2px 8px rgba(0,0,0,0.04), 0 1px 4px rgba(0,0,0,0.04)',
  			medium: '0 4px 12px rgba(0,0,0,0.06), 0 2px 6px rgba(0,0,0,0.08)',
  			prominent: '0 12px 24px rgba(0,0,0,0.08), 0 4px 8px rgba(0,0,0,0.10)'
  		},
  		animation: {
  			shimmer: 'shimmer 2s linear infinite',
  			'accordion-down': 'accordion-down 0.2s ease-out',
  			'accordion-up': 'accordion-up 0.2s ease-out',
  			'spin-slow': 'spin 8s linear infinite'
  		},
  		keyframes: {
  			shimmer: {
  				'100%': {
  					transform: 'translateX(100%)'
  				}
  			},
  			'accordion-down': {
  				from: {
  					height: 0
  				},
  				to: {
  					height: 'var(--radix-accordion-content-height)'
  				}
  			},
  			'accordion-up': {
  				from: {
  					height: 'var(--radix-accordion-content-height)'
  				},
  				to: {
  					height: 0
  				}
  			}
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
};
