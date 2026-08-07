"use client";

import React, { createContext, useContext } from "react";
import { useSample } from "@/app/contexts/SampleContext";
import { type Annotation, type NavAdapter } from "@/types";

const NavAdapterContext = createContext<NavAdapter | null>(null);

export function NavAdapterProvider({
  value,
  children,
}: {
  value: NavAdapter;
  children: React.ReactNode;
}) {
  return (
    <NavAdapterContext.Provider value={value}>
      {children}
    </NavAdapterContext.Provider>
  );
}

export function useNavAdapterOptional(): NavAdapter | null {
  return useContext(NavAdapterContext);
}

export function useNavAdapter(): NavAdapter {
  const navAdapter = useNavAdapterOptional();
  const { annotations, setAnnotations } = useSample();

  if (navAdapter) {
    return navAdapter;
  }

  return {
    getAnnotations: () => annotations,
    afterSave: () => {
      // Mirrors the server-side validation so saved annotator output isn't discarded.
      setAnnotations((previousAnnotations: Annotation[]) =>
        previousAnnotations.map((annotation: Annotation) => ({
          ...annotation,
          validated: true,
        })),
      );
    },
    clear: () => {
      setAnnotations(() => []);
    },
  };
}
