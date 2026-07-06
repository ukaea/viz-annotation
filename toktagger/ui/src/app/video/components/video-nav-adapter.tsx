"use client";

import React from "react";
import { useSample } from "@/app/contexts/SampleContext";
import { NavAdapterProvider } from "@/app/contexts/NavAdapterContext";
import { type Annotation, type NavAdapter } from "@/types";
import { useVideoSession } from "./video-session";

export function VideoNavAdapterBridge({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = useVideoSession();
  const { annotations } = useSample();

  const adapter: NavAdapter = {
    getAnnotations: () => {
      const nowIso = new Date().toISOString();
      return (annotations ?? []).map((annotation): Annotation => {
        if (
          annotation.type === "video_bounding_box" ||
          annotation.type === "video_polygon" ||
          annotation.type === "video_point"
        ) {
          return {
            ...annotation,
            timestamp: annotation.timestamp ?? nowIso,
          };
        }
        return { ...annotation };
      });
    },
    clear: () => {
      session.clearCurrentFrame();
    },
    afterSave: () => {
      session.markSaved();
    },
  };

  return <NavAdapterProvider value={adapter}>{children}</NavAdapterProvider>;
}
