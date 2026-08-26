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
  ComboBox,
  Item,
  Key,
  Heading,
  InlineAlert,
  ToastContainer,
} from "@adobe/react-spectrum";
import { MultiVariateTimeSeriesDataSchema, PlotProps, TaskType } from "@/types";
import { getAnnotationsForSample } from "@/app/core";
import { PeakDetectionTool } from "@/app/components/annotators/peaks";
import { ModelPredictTool } from "@/app/components/tools/modelPredictSample";
import { ShotLabels } from "../annotators/labels";
import { OutlierDetectionTool } from "../annotators/outliers";
import { ChangePointDetectionTool } from "../annotators/changepoints";
import { JumpDetectionTool } from "../annotators/jump";
import { ExportTool } from "./export";
import { ImportButton } from "./import";
import { NavigationBar } from "./nav";
import { useSample } from "@/app/contexts/SampleContext";
import Profile2DThresholdTool from "../annotators/thresholding";
import { Profile2DViewParamsWidget } from "@/app/profile2d/components/profile2dViewParamsWidget";
import { VideoToolbox } from "@/app/video/components/video-toolbox";
import { useServerHealth } from "@/app/contexts/healthContext";

type ColorMapPickerInfo = {
  plotProps: PlotProps;
  setPlotProps: (props: PlotProps) => void;
};

function ColorMapPicker({ plotProps, setPlotProps }: ColorMapPickerInfo) {
  const options = [
    { id: 1, name: "Viridis" },
    { id: 2, name: "Plasma" },
    { id: 3, name: "Inferno" },
    { id: 4, name: "Magma" },
    { id: 5, name: "Cividis" },
  ];

  const onColorMapChange = (key: Key | null) => {
    if (key) {
      const selectedColorMap = Number(key.toString());
      const value = options.find((item) => item.id === selectedColorMap);
      setPlotProps({ ...plotProps, colorMap: value?.name || "Cividis" });
    }
  };

  return (
    <ComboBox
      label="Color Map"
      defaultItems={options}
      inputValue={plotProps.colorMap || "Cividis"}
      onSelectionChange={onColorMapChange}
    >
      {(item) => <Item key={item.id}>{item.name}</Item>}
    </ComboBox>
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
  const {
    project,
    sample,
    data,
    syncAnnotationsFromServer,
    plotProps,
    setPlotProps,
    isValidated,
    canAnnotate,
  } = useSample();

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

  if (data && project.task == TaskType.TimeSeries) {
    const result = MultiVariateTimeSeriesDataSchema.safeParse(data);

    if (!result.success) {
      console.warn("Time series data is not available");
      return;
    }

    const tsData = result.data;
    const labels = project.shot_labels || ["Valid Shot", "Invalid Shot"];
    tools.push({
      name: "Shot Labels",
      component: (
        <ShotLabels labels={labels} canAnnotate={canAnnotate}></ShotLabels>
      ),
    });

    // The automatic annotators exist only to write annotations, and each POSTs to
    // /annotator/* from an effect as soon as it is enabled -- which for a sample that
    // already holds its suggestions happens on mount. Disabling the controls would
    // not stop that, so they are left out entirely for a viewer.
    if (canAnnotate) {
      tools.push({
        name: "Peak Detection",
        component: (
          <PeakDetectionTool
            project_id={project_id}
            sample_id={sample_id}
            data={tsData}
          ></PeakDetectionTool>
        ),
      });

      tools.push({
        name: "Outlier Detection",
        component: (
          <OutlierDetectionTool
            project_id={project_id}
            sample_id={sample_id}
            data={tsData}
          ></OutlierDetectionTool>
        ),
      });

      tools.push({
        name: "Change Point Detection",
        component: (
          <ChangePointDetectionTool
            project_id={project_id}
            sample_id={sample_id}
            data={tsData}
          ></ChangePointDetectionTool>
        ),
      });

      tools.push({
        name: "Jump Detection",
        component: (
          <JumpDetectionTool
            project_id={project_id}
            sample_id={sample_id}
            data={tsData}
          ></JumpDetectionTool>
        ),
      });
    }
  } else if (project.task == TaskType.Profile2D) {
    // Not gated on data so the signal picker below still lets the user recover.
    const labels = project.shot_labels || ["Valid Shot", "Invalid Shot"];
    tools.push({
      name: "Shot Labels",
      component: (
        <ShotLabels labels={labels} canAnnotate={canAnnotate}></ShotLabels>
      ),
    });

    tools.push({
      name: "View Parameters",
      component: <Profile2DViewParamsWidget />,
    });

    tools.push({
      name: "Color Map",
      component: (
        <ColorMapPicker plotProps={plotProps} setPlotProps={setPlotProps} />
      ),
    });

    if (canAnnotate) {
      tools.push({
        name: "Threshold",
        component: (
          <Profile2DThresholdTool
            project_id={project_id}
            sample_id={sample_id}
          />
        ),
      });
    }
  } else if (data && project.task === TaskType.Video) {
    const labels = project.shot_labels || ["Valid Shot", "Invalid Shot"];

    tools.push({
      name: "Shot Labels",
      component: <ShotLabels labels={labels} canAnnotate={canAnnotate} />,
    });

    tools.push({
      name: "Video Tools",
      component: <VideoToolbox />,
      defaultExpanded: true,
    });
  }
  // Predicting writes annotations, so it needs the annotator role.
  if (modelsEnabled && canAnnotate) {
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
    syncAnnotationsFromServer(
      await getAnnotationsForSample(project_id, sample_id),
    );
  };

  return (
    // 100% (not 100vh) so the toolbar stops at the bottom of the window rather
    // than running on below it by the height of the top bar.
    <Provider theme={defaultTheme} height="100%">
      <ToastContainer placement="top" />
      <View overflow="auto" height="100%" width="18vw" flexShrink={0}>
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
                    canAnnotate={canAnnotate}
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
