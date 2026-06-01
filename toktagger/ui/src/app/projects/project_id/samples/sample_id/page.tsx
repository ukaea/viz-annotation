"use client";

import React from "react";
import { Breadcrumbs, Item, Flex } from "@adobe/react-spectrum";
import { Project, Sample, TaskType } from "@/types";
import { TimeSeriesView } from "@/app/time_series/components/time-series";
// import { SpectrogramView } from "@/app/spectrogram/components/spectrogram";
import ToolBar from "@/app/components/tools/toolbar";
import { useParams } from "react-router-dom";
import ErrorView from "@/app/views/error";
import LoadingView from "@/app/views/loading";
import { SampleProvider, useSample } from "@/app/contexts/SampleContext";
import { VideoProviders, VideoView } from "@/app/video/components/video-view";
import { SampleHistoryProvider } from "@/app/contexts/SampleHistoryContext";

type SampleDataBreadCrumbsInfo = {
  project: Project;
  sample: Sample;
};

const SampleDataBreadCrumbs = ({
  project,
  sample,
}: SampleDataBreadCrumbsInfo) => (
  <Breadcrumbs>
    <Item key="projects" href={`/ui/projects`}>
      Projects
    </Item>
    <Item key="project" href={`/ui/projects/${project._id}`}>
      Project: {project.name}
    </Item>
    <Item key="samples">Shot: {sample.shot_id}</Item>
  </Breadcrumbs>
);

const SampleView = () => {
  const { project, error, isLoading, data } = useSample();
  if (!project) return null;
  if (error) return <ErrorView message={error} />;

  if (project.task === TaskType.TimeSeries)
    return isLoading ? <LoadingView /> : <TimeSeriesView />;
  if (project.task === TaskType.Video)
    return isLoading && !data ? <LoadingView /> : <VideoView />;
  // if (project.task === TaskType.Spectrogram) return <SpectrogramView />;
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

  return (
    <div>
      <SampleDataBreadCrumbs project={project} sample={sample} />
      <Flex>
        <SampleTaskProviders>
          <ToolBar />
          <SampleView />
        </SampleTaskProviders>
      </Flex>
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
