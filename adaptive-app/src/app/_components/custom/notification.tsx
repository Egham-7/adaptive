"use client";
import { useEffect } from "react";

type NotificationProps = {
  type: "success" | "error" | "info" | "warning";
  message: string;
  onClose: () => void;
  autoClose?: boolean;
  duration?: number;
};

export default function Notification({
  type,
  message,
  onClose,
  autoClose = true,
  duration = 5000,
}: NotificationProps) {
  useEffect(() => {
    if (autoClose) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [autoClose, duration, onClose]);

  const getStyles = () => {
    switch (type) {
      case "success":
        return "bg-green-500 text-white";
      case "error":
        return "bg-red-500 text-white";
      case "warning":
        return "bg-yellow-500 text-white";
      case "info":
        return "bg-blue-500 text-white";
      default:
        return "bg-gray-500 text-white";
    }
  };

  return (
    <div
      className={`fixed top-4 right-4 z-50 rounded-lg px-6 py-4 shadow-lg ${getStyles()}`}
    >
      <div className="flex items-center justify-between">
        <span>{message}</span>
        <button
          onClick={onClose}
          className="ml-4 text-white hover:text-gray-200"
        >
          ✕
        </button>
      </div>
    </div>
  );
}
