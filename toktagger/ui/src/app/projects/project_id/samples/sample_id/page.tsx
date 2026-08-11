"use client";

import React from "react";
import {
  Provider,
  defaultTheme,
  Breadcrumbs,
  Item,
  ToastContainer,
  Flex,
  View,
} from "@adobe/react-spectrum";
import { Project, Sample, TaskType } from "@/types";
import { TimeSeriesView } from "@/app/time_series/components/time-series";
import { Profile2dView } from "@/app/profile2d/components/profile2d";
import ToolBar from "@/app/components/tools/toolbar";
import { useHref, useNavigate, useParams } from "react-router-dom";
import ErrorView from "@/app/views/error";
import LoadingView from "@/app/views/loading";
import { SampleProvider, useSample } from "@/app/contexts/SampleContext";
import { VideoProviders, VideoView } from "@/app/video/components/video-view";
import { SampleHistoryProvider } from "@/app/contexts/SampleHistoryContext";
import { ModelTrainModal } from "@/app/components/tools/modelTrain";
import { useServerHealth } from "@/app/contexts/healthContext";

type SampleDataBreadCrumbsInfo = {
  project: Project;
  sample: Sample;
};

const SampleDataBreadCrumbs = ({
  project,
  sample,
}: SampleDataBreadCrumbsInfo) => {
  const navigate = useNavigate();
  return (
    <Provider theme={defaultTheme} router={{ navigate, useHref }}>
      <Breadcrumbs>
        <Item key="projects" href={`/ui/projects`}>
          Projects
        </Item>
        <Item key="project" href={`/ui/projects/${project._id}`}>
          Project: {project.name}
        </Item>
        <Item key="samples">Shot: {sample.shot_id}</Item>
      </Breadcrumbs>
    </Provider>
  );
};

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
  const { modelsEnabled } = useServerHealth();

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
      <Provider theme={defaultTheme}>
        <ToastContainer placement="top" />
        <SampleDataBreadCrumbs project={project} sample={sample} />
        <View position="fixed" top="size-100" right="size-100" zIndex={9999}>
          <ModelTrainModal project={project} isEnabled={modelsEnabled} />
        </View>
        <Flex>
          <SampleTaskProviders>
            <ToolBar />
            <SampleView />
          </SampleTaskProviders>
        </Flex>
      </Provider>
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
