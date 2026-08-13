"use client";

import React, { createContext, useContext } from "react";
import { useSample } from "@/app/contexts/SampleContext";
import { useAuth } from "@/app/contexts/AuthContext";
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
  const { user } = useAuth();

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
      // Only clear the current user's own annotations from the local view - other
      // users' annotations (visible via "show others") were never this user's to
      // delete, and the server never touches them on save either. "manual" is the
      // client-side placeholder for an annotation not yet round-tripped through a
      // save, so it always belongs to whoever is currently drawing - the server
      // only stamps the real username once it's saved.
      setAnnotations((previousAnnotations: Annotation[]) =>
        previousAnnotations.filter(
          (annotation) =>
            annotation.created_by !== user?.username &&
            annotation.created_by !== "manual",
        ),
      );
    },
  };
}
