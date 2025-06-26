"use client";

import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import React, { useCallback, useEffect, useState } from "react";

interface Logo {
  name: string;
  id: number;
  img: string;
}

interface LogoColumnProps {
  logos: Logo[];
  index: number;
  currentTime: number;
}

const shuffleArray = <T,>(array: T[]): T[] => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = shuffled[i] as T;
    shuffled[i] = shuffled[j] as T;
    shuffled[j] = temp;
  }
  return shuffled;
};

const distributeLogos = (allLogos: Logo[], columnCount: number): Logo[][] => {
  const shuffled = shuffleArray(allLogos);
  const columns: Logo[][] = Array.from({ length: columnCount }, () => []);

  shuffled.filter(Boolean).forEach((logo, index) => {
    const colIndex = index % columnCount;
    if (!columns[colIndex]) columns[colIndex] = [];
    columns[colIndex].push(logo);
  });

  const maxLength = Math.max(...columns.map((col) => col.length));
  for (const col of columns) {
    while (col.length < maxLength) {
      const randomIndex = Math.floor(Math.random() * shuffled.length);
      const randomLogo = shuffled[randomIndex];
      if (randomLogo) {
        col.push(randomLogo);
      }
    }
  }

  return columns;
};

const LogoColumn: React.FC<LogoColumnProps> = React.memo(
  ({ logos, index, currentTime }) => {
    const cycleInterval = 2000;
    const columnDelay = index * 200;
    const adjustedTime =
      (currentTime + columnDelay) % (cycleInterval * logos.length);
    const currentIndex = Math.floor(adjustedTime / cycleInterval);

    const currentLogo = logos[currentIndex];
    if (!currentLogo) return null;
    const CurrentLogo = currentLogo.img;

    return (
      <motion.div
        className="relative h-14 w-24 overflow-hidden md:h-24 md:w-48"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          delay: index * 0.1,
          duration: 0.5,
          ease: "easeOut",
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={`${currentLogo.id}-${currentIndex}`}
            className="absolute inset-0 flex items-center justify-center"
            initial={{ y: "10%", opacity: 0, filter: "blur(8px)" }}
            animate={{
              y: "0%",
              opacity: 1,
              filter: "blur(0px)",
              transition: {
                type: "spring",
                stiffness: 300,
                damping: 20,
                mass: 1,
                bounce: 0.2,
                duration: 0.5,
              },
            }}
            exit={{
              y: "-20%",
              opacity: 0,
              filter: "blur(6px)",
              transition: {
                type: "tween",
                ease: "easeIn",
                duration: 0.3,
              },
            }}
          >
            <Image
              src={CurrentLogo}
              alt={currentLogo.name}
              width={64}
              height={64}
              className="h-16 w-16 object-contain"
            />
          </motion.div>
        </AnimatePresence>
      </motion.div>
    );
  }
);

interface LogoCarouselProps {
  columnCount?: number;
  logos: Logo[];
}

import { useReducedMotion } from "framer-motion";

export function LogoCarousel({ columnCount = 2, logos }: LogoCarouselProps) {
  const [logoSets, setLogoSets] = useState<Logo[][]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  const updateTime = useCallback(() => {
    if (isPaused || shouldReduceMotion) return;
    setCurrentTime((prevTime) => prevTime + 100);
  }, [isPaused, shouldReduceMotion]);

  useEffect(() => {
    const intervalId = setInterval(updateTime, 100);
    return () => clearInterval(intervalId);
  }, [updateTime]);

  useEffect(() => {
    const distributedLogos = distributeLogos(logos, columnCount);
    setLogoSets(distributedLogos);
  }, [logos, columnCount]);

  return (
    <section
      className="flex space-x-4"
      aria-label="Logo carousel"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      {logoSets.map((logos, index) => (
        <LogoColumn
          key={`column-${index}-${logos[0]?.id ?? "empty"}`}
          logos={logos}
          index={index}
          currentTime={currentTime}
        />
      ))}
    </section>
  );
}

export { LogoColumn };
