"use client";

import React from "react";
import { Annotorious } from "@annotorious/react";
import { ImageDataSchema } from "@/types";
import { Button } from "@adobe/react-spectrum";
import {
  VideoSessionProvider,
  useVideoSession,
} from "@/app/video/components/video-session";
import { FrameAnnotatorHost } from "@/app/video/components/frame-annotator-host";
import { VideoAnnotationToolbar } from "@/app/video/components/video-annotation-toolbar";
import { FrameJumpField } from "@/app/video/components/ui_elements";
import { useSample } from "@/app/contexts/SampleContext";
import { VideoNavAdapterBridge } from "@/app/video/components/video-nav-adapter";
import {
  VideoUiStateProvider,
  useVideoUiState,
} from "@/app/video/components/video-context";
import { useParams } from "react-router-dom";

/**
 * Frame annotator UI wrapper:
 * - Renders the frame navigation (prev/next + jump)
 * - Shows the current frame index (from session state)
 * - Flushes the current frame overlay before frame changes
 *
 * Note: this component does not fetch frames. The parent drives frame changes by
 * updating dataParams and passing the new image when the backend responds.
 */
function VideoFrameAnnotator(props: {
  imageBase64: string;
  goToFrame: (n: number) => void;
}) {
  const session = useVideoSession();
  const { videoFrameBounds } = useSample();

  const prevDisabled =
    session.frame <= 0 ||
    (videoFrameBounds.min !== null && session.frame <= videoFrameBounds.min);
  const nextDisabled =
    videoFrameBounds.max !== null && session.frame >= videoFrameBounds.max;

  const handlePrev = () => {
    if (prevDisabled) return;
    const prev = Math.max(0, session.frame - 1);
    session.flushCurrentFrameOverlay();
    props.goToFrame(prev);
  };

  const handleNext = () => {
    if (nextDisabled) return;
    const next = session.frame + 1;

    session.flushCurrentFrameOverlay();

    // Forward-propagate current annotations into the next frame if that frame is empty.
    // This updates SampleContext working annotations; image loading is driven by goToFrame().
    if (session.propagate) session.forwardPropToNextIfEmpty(next);

    props.goToFrame(next);
  };

  const handleJump = (n: number) => {
    const target = Math.max(0, Math.trunc(n));
    session.flushCurrentFrameOverlay();
    props.goToFrame(target);
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-col items-center gap-2">
        <div className="flex justify-center">
          <div className="flex items-start gap-2">
            <Button
              variant="primary"
              onPress={handlePrev}
              isDisabled={prevDisabled}
            >
              Prev
            </Button>
            <FrameJumpField frame={session.frame} onJump={handleJump} />
            <Button
              variant="primary"
              onPress={handleNext}
              isDisabled={nextDisabled}
            >
              Next
            </Button>
          </div>
        </div>
      </div>

      {/* Frame annotator canvas (image + Annotorious overlay). */}
      <div className="w-full flex flex-col items-center gap-3">
        <FrameAnnotatorHost imageBase64={props.imageBase64} />
      </div>
    </div>
  );
}

export function VideoProviders({ children }: { children: React.ReactNode }) {
  return (
    <VideoUiStateProvider>
      <VideoProvidersInner>{children}</VideoProvidersInner>
    </VideoUiStateProvider>
  );
}

function VideoProvidersInner({ children }: { children: React.ReactNode }) {
  const { project_id, sample_id } = useParams();
  const { data } = useSample();
  const { videoPropagate, setVideoPropagate } = useVideoUiState();

  if (!project_id || !sample_id || !data) {
    return <>{children}</>;
  }

  return (
    <Annotorious>
      <VideoSessionProvider
        key={`${project_id}:${sample_id}`}
        projectId={project_id}
        sampleId={sample_id}
        propagate={videoPropagate}
        setPropagate={setVideoPropagate}
      >
        <VideoNavAdapterBridge>{children}</VideoNavAdapterBridge>
      </VideoSessionProvider>
    </Annotorious>
  );
}

/**
 * Render the image view and wire "go to frame" through SampleContext.
 * This component assumes it is already inside a VideoSessionProvider.
 */
export function VideoView() {
  const { data, dataParams, setDataParams } = useSample();

  if (!data) return null;

  const parsed = ImageDataSchema.safeParse(data);
  if (!parsed.success) {
    throw new Error("Invalid data for UFO view (expected ImageData)");
  }

  const imageBase64 = parsed.data.values;

  /**
   * Request a specific frame from the backend by updating dataParams.
   * The host page owns the fetch and is responsible for updating the session frame
   * once the backend response arrives.
   */
  const goToFrame = (n: number) => {
    if (!Number.isFinite(n)) return;
    const target = Math.max(0, Math.trunc(n));

    setDataParams({
      ...dataParams,
      name: "image",
      frame: target,
    });
  };

  return (
    <div className="min-w-0 flex-1">
      <div className="flex w-full justify-between">
        <div className="min-w-0 flex-1 px-4 py-3">
          <VideoFrameAnnotator
            imageBase64={imageBase64}
            goToFrame={goToFrame}
          />
        </div>
        <VideoAnnotationToolbar />
      </div>
    </div>
  );
}
