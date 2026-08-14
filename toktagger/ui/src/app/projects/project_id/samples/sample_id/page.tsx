"use client";

import React from "react";
import { TaskType } from "@/types";
import { TimeSeriesView } from "@/app/time_series/components/time-series";
import { Profile2dView } from "@/app/profile2d/components/profile2d";
import ToolBar from "@/app/components/tools/toolbar";
import { useParams } from "react-router-dom";
import ErrorView from "@/app/views/error";
import LoadingView from "@/app/views/loading";
import { SampleProvider, useSample } from "@/app/contexts/SampleContext";
import { VideoProviders, VideoView } from "@/app/video/components/video-view";
import { SampleHistoryProvider } from "@/app/contexts/SampleHistoryContext";
import { useBreadcrumbs } from "@/app/contexts/BreadcrumbContext";

const SampleView = () => {
  const { project, error, isLoading, data } = useSample();
  if (!project) return null;
  if (error) return <ErrorView message={error} />;

  if (project.task === TaskType.TimeSeries)
    return isLoading ? <LoadingView /> : <TimeSeriesView />;
  if (project.task === TaskType.Video)
    return isLoading && !data ? <LoadingView /> : <VideoView />;
  if (project.task === TaskType.Profile2D)
    return isLoading ? <LoadingView /> : <Profile2dView />;
  return null;
};

function SampleTaskProviders({ children }: { children: React.ReactNode }) {
  const { project } = useSample();

  if (project?.task === TaskType.Video) {
    return <VideoProviders>{children}</VideoProviders>;
  }

  return <>{children}</>;
}

function SamplePageContent(props: { sampleId: string }) {
  const { project, sample, isLoading, error } = useSample();
  useBreadcrumbs(
    project && sample
      ? [
          { key: "projects", label: "Projects", href: "/ui/projects" },
          {
            key: "project",
            label: `Project: ${project.name}`,
            href: `/ui/projects/${project._id}`,
          },
          { key: "samples", label: `Shot: ${sample.shot_id}` },
        ]
      : [{ key: "projects", label: "Projects", href: "/ui/projects" }],
  );

  // Early returns AFTER all hooks
  if (error) return <ErrorView message={error} />;

  if (!project) {
    return isLoading ? (
      <LoadingView />
    ) : (
      <ErrorView message="Project not found." />
    );
  }

  if (!sample) {
    return isLoading ? (
      <LoadingView />
    ) : (
      <ErrorView message="Sample not found." />
    );
  }

  //  Prevent a stale render during route param transitions
  if (sample._id !== props.sampleId) {
    return <LoadingView />;
  }

  // h-full/min-h-0 hand the height of the area below the top bar down to the
  // toolbar and the view, so each can scroll internally instead of the whole
  // page growing past the bottom of the window.
  return (
    <div className="flex h-full min-h-0">
      <SampleTaskProviders>
        <ToolBar />
        <SampleView />
      </SampleTaskProviders>
    </div>
  );
}

export default function SamplePage() {
  const { project_id, sample_id } = useParams();

  if (!project_id || !sample_id) return null;

  return (
    <SampleProvider projectId={project_id} sampleId={sample_id}>
      <SampleHistoryProvider projectId={project_id}>
        <SamplePageContent sampleId={sample_id} />
      </SampleHistoryProvider>
    </SampleProvider>
  );
}
