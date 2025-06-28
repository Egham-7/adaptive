"use client";
import { verifySession } from "@/actions/verify-session";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import Notification from "../custom/notification";

type Props = {
  children: React.ReactNode;
};

// Safe sessionStorage access
const getSessionStorage = (key: string): string | null => {
  try {
    if (typeof window !== "undefined" && window.sessionStorage) {
      return sessionStorage.getItem(key);
    }
    return null;
  } catch {
    return null;
  }
};

const setSessionStorage = (key: string, value: string): void => {
  try {
    if (typeof window !== "undefined" && window.sessionStorage) {
      sessionStorage.setItem(key, value);
    }
  } catch {
    // Ignore sessionStorage errors
  }
};

export default function PaymentNotificationWrapper({ children }: Props) {
  const searchParams = useSearchParams();
  const [notification, setNotification] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  useEffect(() => {
    const success = searchParams.get("success");
    const canceled = searchParams.get("canceled");
    const sessionId = searchParams.get("session_id");

    async function handleSuccess() {
      if (sessionId) {
        const shownNotification = getSessionStorage(
          `notification_shown_${sessionId}`,
        );
        if (shownNotification) {
          return;
        }

        try {
          const { isValid } = await verifySession(sessionId);

          if (isValid) {
            setNotification({
              type: "success",
              message: "Payment successful! Welcome to Pro!",
            });
            setSessionStorage(`notification_shown_${sessionId}`, "true");
          } else {
            setNotification({
              type: "error",
              message: "Payment verification failed. Please contact support.",
            });
          }
        } catch (error) {
          console.error("Error verifying session:", error);
          setNotification({
            type: "error",
            message: "Payment verification failed. Please contact support.",
          });
        }
      }
    }

    async function handleCancel() {
      if (sessionId) {
        const shownNotification = getSessionStorage(
          `notification_shown_${sessionId}`,
        );
        if (shownNotification) {
          return;
        }

        setNotification({
          type: "error",
          message: "Payment was canceled. You can try again anytime.",
        });
        setSessionStorage(`notification_shown_${sessionId}`, "true");
      }
    }

    if (success === "true") {
      handleSuccess();
    } else if (canceled === "true") {
      handleCancel();
    }
  }, [searchParams]);

  return (
    <>
      {notification && (
        <Notification
          type={notification.type}
          message={notification.message}
          onClose={() => setNotification(null)}
        />
      )}
      {children}
    </>
  );
}
