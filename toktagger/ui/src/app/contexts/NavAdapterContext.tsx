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
      // saveSampleAnnotations validates on the server, so mirror that locally.
      // Annotator output is only kept when it is validated, so without this an
      // annotator's annotations would vanish on toggling it off despite being saved.
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
