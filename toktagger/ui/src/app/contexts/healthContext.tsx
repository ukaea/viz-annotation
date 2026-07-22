"use client";
import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from "react";
import { BACKEND_API_URL } from "@/app/core";
import { HealthInfo } from "@/types";

interface ServerHealthContextType {
  version: string;
  dbConnected: boolean;
  modelsEnabled: boolean;
  isLoading: boolean;
  gpuAvailable: boolean;
  error: string | null;
}

const ServerHealthContext = createContext<ServerHealthContextType | undefined>(
  undefined,
);

async function getHealth(): Promise<HealthInfo> {
  const response = await fetch(`${BACKEND_API_URL}/health`);
  if (!response.ok) {
    throw new Error(
      `Failed to contact TokTagger server: ${response.statusText}`,
    );
  }
  const payload = await response.json();
  return payload as HealthInfo;
}

export function ServerHealthProvider({ children }: { children: ReactNode }) {
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealthInfo = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const data = await getHealth();
        setHealthInfo(data);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);
        console.error("Error fetching schema:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHealthInfo();
  }, []);

  const value: ServerHealthContextType = {
    version: healthInfo?.name ?? "",
    dbConnected: healthInfo?.db_connected ?? false,
    modelsEnabled: healthInfo?.models_enabled ?? false,
    gpuAvailable: healthInfo?.gpu_available ?? false,
    isLoading: isLoading,
    error: error,
  };

  return (
    <ServerHealthContext.Provider value={value}>
      {children}
    </ServerHealthContext.Provider>
  );
}

export function useServerHealth() {
  const ctx = useContext(ServerHealthContext);
  if (!ctx) {
    throw new Error(
      "useServerHealth must be used inside a ServerHealthProvider",
    );
  }
  return ctx;
}
