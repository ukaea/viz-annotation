"use client";

import React, { useEffect } from "react";
import { Annotorious } from "@annotorious/react";
import { ImageDataSchema } from "@/types";
import {
  ActionButton,
  Button,
  Flex,
  Grid,
  ProgressCircle,
  Text,
} from "@adobe/react-spectrum";
import Label from "@spectrum-icons/workflow/Label";
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
} from "@/app/contexts/VideoContext";
import { useParams } from "react-router-dom";

import { AnnotationPopup } from "@/app/video/components/annotation-popup";

function FrameLabelBadge() {
  const { frame, frameLabels, hideAnnotations, editMode, removeFrameLabel } =
    useVideoSession();
  const [isPopupOpen, setIsPopupOpen] = React.useState(false);

  useEffect(() => {
    setIsPopupOpen(false);
  }, [frame, frameLabels, hideAnnotations]);

  if (hideAnnotations || frameLabels.length === 0) return null;

  const frameLabelNames = frameLabels
    .map((frameLabel) => frameLabel.label)
    .join(", ");

  return (
    <Flex position="relative">
      <ActionButton
        aria-label={`Frame labels: ${frameLabelNames}`}
        onPress={() => setIsPopupOpen((prev) => !prev)}
      >
        <Label aria-hidden="true" />
        <Text>{frameLabelNames}</Text>
      </ActionButton>

      {isPopupOpen && (
        <Flex
          position="absolute"
          zIndex={60}
          direction="column"
          gap="size-100"
          left={0}
          top="size-500"
        >
          {frameLabels.map((frameLabel) => (
            <AnnotationPopup
              key={`${frameLabel.label}::${frameLabel.track_id}`}
              className={frameLabel.label}
              trackId={frameLabel.track_id}
              heading="Frame label"
              details={`Frame ${frame} · ${frameLabel.created_by}`}
              deleteDisabled={!editMode}
              onDeleteBox={() => {
                if (frameLabels.length <= 1) {
                  setIsPopupOpen(false);
                }
                removeFrameLabel(frameLabel.label, frameLabel.track_id);
              }}
              onClose={() => setIsPopupOpen(false)}
            />
          ))}
        </Flex>
      )}
    </Flex>
  );
}

/**
 * Frame annotator UI wrapper:
 * - Renders the frame navigation (prev/next + jump)
 * - Shows the latest requested frame index
 * - Flushes the current frame overlay before frame changes
 *
 * Note: this component does not fetch frames. The parent drives frame changes by
 * updating dataParams and passing the new image when the backend responds.
 */
function VideoFrameAnnotator(props: {
  imageBase64: string;
  desiredFrame: number;
  goToFrame: (n: number) => boolean;
  goToRelativeFrame: (delta: number, fallbackFrame: number) => number | null;
}) {
  const session = useVideoSession();
  const { videoFrameBounds, isLoading } = useSample();
  const isFramePending = isLoading && props.desiredFrame !== session.frame;

  const prevDisabled =
    props.desiredFrame <= 0 ||
    (videoFrameBounds.min !== null &&
      props.desiredFrame <= videoFrameBounds.min);
  const nextDisabled =
    videoFrameBounds.max !== null && props.desiredFrame >= videoFrameBounds.max;

  const handlePrev = () => {
    if (prevDisabled) return;
    const prev = props.goToRelativeFrame(-1, session.frame);
    if (prev === null) return;
    session.flushCurrentFrameOverlay();
  };

  const handleNext = () => {
    if (nextDisabled) return;
    const next = props.goToRelativeFrame(1, session.frame);
    if (next === null) return;

    session.flushCurrentFrameOverlay();

    // Forward-propagate current annotations into the next frame if that frame is empty.
    // This updates SampleContext working annotations; image loading is driven by goToFrame().
    if (session.propagate) session.forwardPropToNextIfEmpty(next);
  };

  const handleJump = (n: number) => {
    const target = Math.max(0, Math.trunc(n));
    if (!props.goToFrame(target)) return;
    session.flushCurrentFrameOverlay();
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <Grid
        width="100%"
        maxWidth="1100px"
        columns={["1fr", "auto", "1fr"]}
        areas={["badge navigation spacer"]}
        alignItems="start"
      >
        <Flex gridArea="badge" justifyContent="center">
          <FrameLabelBadge />
        </Flex>
        <Flex gridArea="navigation" alignItems="start" gap="size-100">
          <Button
            variant="primary"
            onPress={handlePrev}
            isDisabled={prevDisabled}
          >
            Prev
          </Button>
          <FrameJumpField frame={props.desiredFrame} onJump={handleJump} />
          <Button
            variant="primary"
            onPress={handleNext}
            isDisabled={nextDisabled}
          >
            Next
          </Button>
        </Flex>
        <Flex gridArea="spacer" />
      </Grid>

      {/* Frame annotator canvas (image + Annotorious overlay). */}
      <div className="relative w-full flex flex-col items-center gap-3">
        <FrameAnnotatorHost imageBase64={props.imageBase64} />
        {isFramePending && (
          <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center">
            <ProgressCircle
              aria-label={`Loading frame ${props.desiredFrame}`}
              size="L"
              isIndeterminate
            />
          </div>
        )}
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
  const desiredFrameRef = React.useRef<number | null>(null);

  useEffect(() => {
    if (
      dataParams.name === "image" &&
      typeof dataParams.frame === "number" &&
      Number.isFinite(dataParams.frame)
    ) {
      desiredFrameRef.current = dataParams.frame;
      return;
    }

    const currentData = ImageDataSchema.safeParse(data);
    if (currentData.success) {
      desiredFrameRef.current = currentData.data.frame;
    }
  }, [data, dataParams]);

  if (!data) return null;

  const parsed = ImageDataSchema.safeParse(data);
  if (!parsed.success) {
    throw new Error("Invalid data for UFO view (expected ImageData)");
  }

  const imageBase64 = parsed.data.values;
  const desiredFrame =
    dataParams.name === "image" &&
    typeof dataParams.frame === "number" &&
    Number.isFinite(dataParams.frame)
      ? dataParams.frame
      : parsed.data.frame;

  /**
   * Request a specific frame from the backend by updating dataParams.
   * The host page owns the fetch and is responsible for updating the session frame
   * once the backend response arrives.
   */
  const goToFrame = (n: number): boolean => {
    if (!Number.isFinite(n)) return false;
    const target = Math.max(0, Math.trunc(n));
    const currentDesiredFrame = desiredFrameRef.current ?? desiredFrame;
    if (target === currentDesiredFrame) return false;

    desiredFrameRef.current = target;

    setDataParams((prev) => {
      if (prev.name === "image" && prev.frame === target) return prev;

      return {
        ...prev,
        name: "image",
        frame: target,
      };
    });

    return true;
  };

  const goToRelativeFrame = (
    delta: number,
    fallbackFrame: number,
  ): number | null => {
    if (!Number.isFinite(delta)) return null;

    const currentDesiredFrame =
      desiredFrameRef.current ?? Math.max(0, Math.trunc(fallbackFrame));
    const target = Math.max(
      0,
      Math.trunc(currentDesiredFrame + Math.trunc(delta)),
    );
    if (target === currentDesiredFrame) return null;

    desiredFrameRef.current = target;

    setDataParams((prev) => {
      const previousDesiredFrame =
        prev.name === "image" &&
        typeof prev.frame === "number" &&
        Number.isFinite(prev.frame)
          ? prev.frame
          : Math.max(0, Math.trunc(fallbackFrame));
      const nextFrame = Math.max(
        0,
        Math.trunc(previousDesiredFrame + Math.trunc(delta)),
      );

      if (nextFrame === previousDesiredFrame) return prev;

      return {
        ...prev,
        name: "image",
        frame: nextFrame,
      };
    });

    return target;
  };

  return (
    <div className="min-w-0 flex-1">
      <div className="flex w-full justify-between">
        <div className="min-w-0 flex-1 px-4 py-3">
          <VideoFrameAnnotator
            imageBase64={imageBase64}
            desiredFrame={desiredFrame}
            goToFrame={goToFrame}
            goToRelativeFrame={goToRelativeFrame}
          />
        </div>
        <VideoAnnotationToolbar desiredFrame={desiredFrame} />
      </div>
    </div>
  );
}
