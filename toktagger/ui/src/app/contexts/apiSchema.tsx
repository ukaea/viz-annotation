"use client";
import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import { BACKEND_API_URL, apiFetch } from "@/app/core";

interface APISchemaContextType {
  schema: Record<string, unknown> | null;
  isLoading: boolean;
  error: string | null;
}

const APISchemaContext = createContext<APISchemaContextType | undefined>(
  undefined,
);

interface APISchemaProviderProps {
  children: ReactNode;
}

async function getAPISchema(): Promise<Record<string, unknown>> {
  const response = await apiFetch(`${BACKEND_API_URL}/openapi.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch schema: ${response.statusText}`);
  }
  const payload = await response.json();
  return payload as Record<string, unknown>;
}

export const APISchemaProvider: React.FC<APISchemaProviderProps> = ({
  children,
}) => {
  const [schema, setSchema] = useState<Record<string, unknown> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSchema = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const data = await getAPISchema();
        setSchema(data);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError(errorMessage);
        console.error("Error fetching schema:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSchema();
  }, []);

  return (
    <APISchemaContext.Provider value={{ schema, isLoading, error }}>
      {children}
    </APISchemaContext.Provider>
  );
};

export const useAPISchema = (): APISchemaContextType => {
  const context = useContext(APISchemaContext);
  if (context === undefined) {
    throw new Error("useAPISchema must be used within an APISchemaProvider");
  }
  return context;
};

export default APISchemaProvider;
