"use client";
import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";

export interface BreadcrumbItem {
  key: string;
  label: string;
  href?: string;
}

interface BreadcrumbContextType {
  items: BreadcrumbItem[];
  setItems: (items: BreadcrumbItem[]) => void;
}

const BreadcrumbContext = createContext<BreadcrumbContextType | undefined>(
  undefined,
);

export function BreadcrumbProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<BreadcrumbItem[]>([]);
  return (
    <BreadcrumbContext.Provider value={{ items, setItems }}>
      {children}
    </BreadcrumbContext.Provider>
  );
}

function useBreadcrumbContext() {
  const ctx = useContext(BreadcrumbContext);
  if (!ctx) {
    throw new Error(
      "Breadcrumb hooks must be used within a BreadcrumbProvider",
    );
  }
  return ctx;
}

export function useBreadcrumbItems() {
  return useBreadcrumbContext().items;
}

export function useBreadcrumbs(items: BreadcrumbItem[]) {
  const { setItems } = useBreadcrumbContext();
  const key = JSON.stringify(items);
  useEffect(() => {
    setItems(items);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
}
