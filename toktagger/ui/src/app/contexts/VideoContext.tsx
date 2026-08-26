"use client";

import React, { createContext, useCallback, useContext, useState } from "react";
import type { ActiveDrawingTool } from "@/app/video/components/types";

type VideoUiStateContextType = {
  videoPropagate: boolean;
  setVideoPropagate: (value: boolean) => void;
  videoLastClassName: string | null;
  setVideoLastClassName: (value: string | null) => void;
  videoEditMode: boolean;
  setVideoEditMode: (value: boolean) => void;
  videoDrawingTool: ActiveDrawingTool;
  setVideoDrawingTool: (value: ActiveDrawingTool) => void;
};

const VideoUiStateContext = createContext<VideoUiStateContextType | undefined>(
  undefined,
);

const videoUiStateSnapshot = {
  videoPropagate: true,
  videoLastClassName: null as string | null,
  videoEditMode: false,
  videoDrawingTool: null as ActiveDrawingTool,
};

export function VideoUiStateProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [videoPropagate, setVideoPropagateState] = useState(
    () => videoUiStateSnapshot.videoPropagate,
  );
  const [videoLastClassName, setVideoLastClassNameState] = useState<
    string | null
  >(() => videoUiStateSnapshot.videoLastClassName);
  const [videoEditMode, setVideoEditModeState] = useState(
    () => videoUiStateSnapshot.videoEditMode,
  );
  const [videoDrawingTool, setVideoDrawingToolState] =
    useState<ActiveDrawingTool>(() => videoUiStateSnapshot.videoDrawingTool);

  const setVideoPropagate = useCallback((value: boolean) => {
    videoUiStateSnapshot.videoPropagate = value;
    setVideoPropagateState(value);
  }, []);

  const setVideoLastClassName = useCallback((value: string | null) => {
    videoUiStateSnapshot.videoLastClassName = value;
    setVideoLastClassNameState(value);
  }, []);

  const setVideoEditMode = useCallback((value: boolean) => {
    videoUiStateSnapshot.videoEditMode = value;
    setVideoEditModeState(value);
  }, []);

  const setVideoDrawingTool = useCallback((value: ActiveDrawingTool) => {
    videoUiStateSnapshot.videoDrawingTool = value;
    setVideoDrawingToolState(value);
  }, []);

  return (
    <VideoUiStateContext.Provider
      value={{
        videoPropagate,
        setVideoPropagate,
        videoLastClassName,
        setVideoLastClassName,
        videoEditMode,
        setVideoEditMode,
        videoDrawingTool,
        setVideoDrawingTool,
      }}
    >
      {children}
    </VideoUiStateContext.Provider>
  );
}

export function useVideoUiState() {
  const ctx = useContext(VideoUiStateContext);
  if (!ctx) {
    throw new Error("useVideoUiState must be used inside VideoUiStateProvider");
  }
  return ctx;
}
