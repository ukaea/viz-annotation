"use client";
import {
  Provider,
  defaultTheme,
  Flex,
  View,
  Header,
  Accordion,
  Disclosure,
  DisclosureTitle,
  DisclosurePanel,
  SearchField,
  Heading,
  InlineAlert,
} from "@adobe/react-spectrum";
import { Annotation, TaskType } from "@/types";
import { useNavigate } from "react-router-dom";

import { BACKEND_API_URL, getAnnotationsForSample } from "@/app/core";
import { useSample } from "@/app/contexts/SampleContext";

import { PeakDetectionTool } from "@/app/components/annotators/peaks";
import { ModelPredictTool } from "@/app/components/tools/modelPredictSample";
import { ShotLabels } from "../annotators/labels";
import { OutlierDetectionTool } from "../annotators/outliers";
import { ChangePointDetectionTool } from "../annotators/changepoints";
import { JumpDetectionTool } from "../annotators/jump";
import { ExportTool } from "./export";
import { ImportButton } from "./import";
import { NavigationBar } from "./nav";
import { Profile2DViewParamsWidget } from "../tools/profile2dViewParamsWidget";

import { ColorMapPicker } from "./colorMapPicker";
import Profile2DThresholdTool from "../annotators/thresholding";
import { useState } from "react";
import { VideoToolbox } from "@/app/video/components/video-toolbox";
import { useServerHealth } from "@/app/contexts/healthContext";

// ------------------------------
// Helpers: backend save + sample navigation
// ------------------------------

async function saveAnnotationsValidated(
  project_id: string,
  sample_id: string,
  annotations: Annotation[],
) {
  const ANNOTATIONS_URL = `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotations`;

  const validatedAnnotations: Annotation[] = annotations.map(
    (annotation: Annotation) => ({
      ...annotation,
      validated: true,
      created_by: "manual",
    }),
  );

  const response = await fetch(ANNOTATIONS_URL, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(validatedAnnotations),
  });
  return response;
}

async function getShotSample(project_id: string, shot_id: string) {
  const NEXT_URL = `${BACKEND_API_URL}/projects/${project_id}/samples?shot_id=${shot_id}`;
  const sampleResult = await fetch(NEXT_URL);
  const sampleArray = await sampleResult.json();
  let sample = null;
  if (sampleArray.length > 0) {
    sample = sampleArray[0];
  }
  return sample;
}

// ------------------------------
// Standard TS/Profile2D controls
// ------------------------------

type SaveInfo = {
  project_id: string;
  sample_id: string;
  annotations: Annotation[];
};

export function ShotSearch({ project_id, sample_id, annotations }: SaveInfo) {
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = useState<string>("");

  const onSearchSubmit = async (newValue: string) => {
    if (newValue == "") {
      setErrorMessage("");
    } else if (/^[0-9]*$/.test(newValue)) {
      setErrorMessage("");
      const shot_id = newValue;
      try {
        const sample = await getShotSample(project_id, shot_id);
        if (sample !== null) {
          await saveAnnotationsValidated(project_id, sample_id, annotations);
          const NEXT_SAMPLE_URL = `/ui/projects/${project_id}/samples/${sample._id}`;
          navigate(NEXT_SAMPLE_URL);
        } else {
          setErrorMessage("Shot not found!");
        }
      } catch (err) {
        console.error("Failed to fetch data:", err);
      }
    } else {
      setErrorMessage("Please enter a number.");
    }
  };

  return (
    <SearchField
      label="Jump to Shot"
      onSubmit={onSearchSubmit}
      validationState={errorMessage ? "invalid" : undefined}
      errorMessage={errorMessage}
    />
  );
}

function AnnotationStatusAlert({ isValidated }: { isValidated: boolean }) {
  return (
    <Flex justifyContent="center" width="100%" marginTop="size-200">
      <InlineAlert
        variant={isValidated ? "positive" : "notice"}
        UNSAFE_style={{
          paddingTop: "5px",
          paddingBottom: "5px",
          paddingLeft: "10px",
          paddingRight: "10px",
        }}
      >
        <Heading>
          {isValidated ? "Annotations Validated" : "Annotations Not Validated"}
        </Heading>
      </InlineAlert>
    </Flex>
  );
}

export default function ToolBar() {
  const { project, sample, setAnnotations, isValidated } = useSample();

  const { modelsEnabled } = useServerHealth();

  if (!project || !sample) {
    console.warn("Project or sample not found in ToolBar");
    return null;
  }

  const project_id = project._id;
  const sample_id = sample._id;

  if (project_id == null || sample_id == null) {
    console.warn("Invalid project_id or sample_id in ToolBar");
    return null;
  }

  const tools: {
    name: string;
    component: React.ReactNode;
    defaultExpanded?: boolean;
  }[] = [];

  const labels = project.shot_labels || ["Valid Shot", "Invalid Shot"];
  tools.push({
    name: "Shot Labels",
    component: <ShotLabels labels={labels}></ShotLabels>,
  });

  if (project.task == TaskType.TimeSeries) {
    tools.push({
      name: "Peak Detection",
      component: (
        <PeakDetectionTool
          project_id={project_id}
          sample_id={sample_id}
        ></PeakDetectionTool>
      ),
    });

    tools.push({
      name: "Outlier Detection",
      component: (
        <OutlierDetectionTool
          project_id={project_id}
          sample_id={sample_id}
        ></OutlierDetectionTool>
      ),
    });

    tools.push({
      name: "Change Point Detection",
      component: (
        <ChangePointDetectionTool
          project_id={project_id}
          sample_id={sample_id}
        ></ChangePointDetectionTool>
      ),
    });

    tools.push({
      name: "Jump Detection",
      component: (
        <JumpDetectionTool
          project_id={project_id}
          sample_id={sample_id}
        ></JumpDetectionTool>
      ),
    });
  } else if (project.task == TaskType.Profile2D) {
    tools.push({
      name: "View Parameters",
      component: <Profile2DViewParamsWidget />,
    });

    tools.push({
      name: "Color Map",
      component: <ColorMapPicker />,
    });

    tools.push({
      name: "Threshold",
      component: (
        <Profile2DThresholdTool project_id={project_id} sample_id={sample_id} />
      ),
    });
  } else if (project.task === TaskType.Video) {
    const labels = project.shot_labels || ["Valid Shot", "Invalid Shot"];

    tools.push({
      name: "Shot Labels",
      component: <ShotLabels labels={labels} />,
    });

    tools.push({
      name: "Video Tools",
      component: <VideoToolbox />,
      defaultExpanded: true,
    });
  }
  if (modelsEnabled) {
    tools.push({
      name: "Model Prediction",
      component: (
        <ModelPredictTool
          project_id={project_id}
          sample_id={sample_id}
        ></ModelPredictTool>
      ),
    });
  }

  const refreshAnnotations = async () => {
    const dbAnnotations = await getAnnotationsForSample(project_id, sample_id);
    setAnnotations(() => dbAnnotations);
  };

  return (
    <Provider theme={defaultTheme} height="100vh">
      <View overflow="auto" height="100vh" width="18vw">
        <Flex
          direction="column"
          alignItems="center"
          justifyContent="center"
          gap="size-100"
          width="100%"
        >
          {isValidated !== null && (
            <AnnotationStatusAlert isValidated={isValidated} />
          )}
          <Flex
            direction="column"
            alignItems="center"
            justifyContent="center"
            gap="size-100"
            width="100%"
          >
            <Header height="size-300" marginBottom="size-100">
              <span style={{ fontSize: "1.2rem" }}>Controls</span>
            </Header>
            <NavigationBar project_id={project_id} sample_id={sample_id} />
            <Accordion allowsMultipleExpanded={true} width="100%">
              <Disclosure>
                <DisclosureTitle>
                  <span style={{ fontSize: "0.8rem" }}>Export Annotations</span>
                </DisclosureTitle>
                <DisclosurePanel>
                  <ExportTool project={project} sample={sample} />
                </DisclosurePanel>
              </Disclosure>
              <Disclosure>
                <DisclosureTitle>
                  <span style={{ fontSize: "0.8rem" }}>Import Annotations</span>
                </DisclosureTitle>
                <DisclosurePanel>
                  <ImportButton
                    project={project}
                    sample={sample}
                    refreshAnnotations={refreshAnnotations}
                  />
                </DisclosurePanel>
              </Disclosure>
            </Accordion>
          </Flex>
          {tools.length > 0 && (
            <>
              <Flex justifyContent="center" alignItems="center">
                <Header height="size-300" marginBottom="size-100">
                  <span style={{ fontSize: "1.2rem" }}>Toolbox</span>
                </Header>
              </Flex>

              <Accordion
                allowsMultipleExpanded={true}
                defaultExpandedKeys={tools
                  .filter((item) => item.defaultExpanded)
                  .map((item) => item.name)}
                width="100%"
              >
                {tools.map((item) => (
                  <Disclosure key={item.name} id={item.name}>
                    <DisclosureTitle>
                      <span style={{ fontSize: "0.8rem" }}>{item.name}</span>
                    </DisclosureTitle>
                    <DisclosurePanel>{item.component}</DisclosurePanel>
                  </Disclosure>
                ))}
              </Accordion>
            </>
          )}
        </Flex>
      </View>
    </Provider>
  );
}
