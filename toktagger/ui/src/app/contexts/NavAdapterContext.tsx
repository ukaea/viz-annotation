"use client";

import React, { createContext, useContext } from "react";
import { useSample } from "@/app/contexts/SampleContext";
import { useAuth } from "@/app/contexts/AuthContext";
import { deleteSampleAnnotations } from "@/app/core";
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
  const { annotations, setAnnotations, project, sample } = useSample();
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
    clear: async (includeOthers?: boolean) => {
      // Clearing what the user can see means clearing other users' annotations and
      // model predictions too. A save cannot do that - its replace step is scoped to
      // the caller's own created_by - so they are deleted here explicitly, and the
      // local view is only emptied once that succeeds.
      if (includeOthers) {
        if (project?._id && sample?._id) {
          await deleteSampleAnnotations(project._id, sample._id);
        }
        setAnnotations(() => []);
        return;
      }

      // "Show others" is off, so the user can only see their own annotations and
      // only those are cleared. They are removed from the local view alone; the save
      // that follows is what deletes them server-side. "manual" is the placeholder
      // used until the auth context resolves, so it belongs to whoever is drawing.
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
