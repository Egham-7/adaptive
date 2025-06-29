"use client";

import { motion } from "framer-motion";

interface AnimatedLogoProps {
  className?: string;
  size?: number;
}

export function AnimatedLogo({ className = "w-20 h-20", size }: AnimatedLogoProps) {
  return (
    <motion.div
      className={`relative mx-auto ${className}`}
      initial={{ scale: 0, rotate: -180 }}
      animate={{ scale: 1, rotate: 0 }}
      transition={{
        duration: 1,
        ease: "easeOut",
        delay: 0.2,
      }}
    >
      <motion.svg
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 200 200"
        width={size || "100%"}
        height={size || "100%"}
        className="w-full h-full"
        animate={{
          rotate: [0, 5, -5, 0],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        <g clipPath="url(#cs_clip_1_ellipse-12)">
          <mask
            id="cs_mask_1_ellipse-12"
            style={{ maskType: "alpha" }}
            width="200"
            height="200"
            x="0"
            y="0"
            maskUnits="userSpaceOnUse"
          >
            <path
              fill="#fff"
              fillRule="evenodd"
              d="M100 150c27.614 0 50-22.386 50-50s-22.386-50-50-50-50 22.386-50 50 22.386 50 50 50zm0 50c55.228 0 100-44.772 100-100S155.228 0 100 0 0 44.772 0 100s44.772 100 100 100z"
              clipRule="evenodd"
            ></path>
          </mask>
          <g mask="url(#cs_mask_1_ellipse-12)">
            <path fill="#fff" d="M200 0H0v200h200V0z"></path>
            <path
              fill="hsl(var(--primary))"
              fillOpacity="0.33"
              d="M200 0H0v200h200V0z"
            ></path>
            <motion.g
              filter="url(#filter0_f_844_2811)"
              animate={{
                opacity: [0.6, 1, 0.6],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            >
              <path
                fill="hsl(var(--primary))"
                d="M110 32H18v68h92V32z"
              ></path>
              <path
                fill="hsl(var(--primary))"
                d="M188-24H15v98h173v-98z"
              ></path>
              <path
                fill="hsl(var(--primary) / 0.8)"
                d="M175 70H5v156h170V70z"
              ></path>
              <path
                fill="hsl(var(--primary) / 0.6)"
                d="M230 51H100v103h130V51z"
              ></path>
            </motion.g>
          </g>
        </g>
        <defs>
          <filter
            id="filter0_f_844_2811"
            width="385"
            height="410"
            x="-75"
            y="-104"
            colorInterpolationFilters="sRGB"
            filterUnits="userSpaceOnUse"
          >
            <feFlood floodOpacity="0" result="BackgroundImageFix"></feFlood>
            <feBlend
              in="SourceGraphic"
              in2="BackgroundImageFix"
              result="shape"
            ></feBlend>
            <feGaussianBlur
              result="effect1_foregroundBlur_844_2811"
              stdDeviation="40"
            ></feGaussianBlur>
          </filter>
          <clipPath id="cs_clip_1_ellipse-12">
            <path fill="#fff" d="M0 0H200V200H0z"></path>
          </clipPath>
        </defs>
        <g
          style={{ mixBlendMode: "overlay" }}
          mask="url(#cs_mask_1_ellipse-12)"
        >
          <path
            fill="gray"
            stroke="transparent"
            d="M200 0H0v200h200V0z"
            filter="url(#cs_noise_1_ellipse-12)"
          ></path>
        </g>
        <defs>
          <filter
            id="cs_noise_1_ellipse-12"
            width="100%"
            height="100%"
            x="0%"
            y="0%"
            filterUnits="objectBoundingBox"
          >
            <feTurbulence
              baseFrequency="0.6"
              numOctaves="5"
              result="out1"
              seed="4"
            ></feTurbulence>
            <feComposite
              in="out1"
              in2="SourceGraphic"
              operator="in"
              result="out2"
            ></feComposite>
            <feBlend
              in="SourceGraphic"
              in2="out2"
              mode="overlay"
              result="out3"
            ></feBlend>
          </filter>
        </defs>
      </motion.svg>
    </motion.div>
  );
}