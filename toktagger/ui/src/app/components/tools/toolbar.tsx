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
} from "@adobe/react-spectrum";
import {
  MultiVariateTimeSeriesDataSchema,
  PlotProps,
  SpectrogramData,
  SpectrogramDataSchema,
  SpectrogramViewParamsSchema,
  TaskType,
  ViewParams,
} from "@/types";
import { getAnnotationsForSample } from "@/app/core";
import { PeakDetectionTool } from "@/app/components/annotators/peaks";
import { DataRangeSlider } from "@/app/components/tools/dataRangeSlider";
import { ModelPredictTool } from "@/app/components/tools/modelPredictSample";
import { ShotLabels } from "../annotators/labels";
import { OutlierDetectionTool } from "../annotators/outliers";
import { ChangePointDetectionTool } from "../annotators/changepoints";
import { JumpDetectionTool } from "../annotators/jump";
import { ExportTool } from "./export";
import { ImportButton } from "./import";
import { NavigationBar } from "./nav";
import { useSample } from "@/app/contexts/SampleContext";
import SpectrogramThresholdTool from "../annotators/thresholding";
import { VideoToolbox } from "@/app/video/components/video-toolbox";
import { useServerHealth } from "@/app/contexts/healthContext";

type AmplitudeSliderInfo = {
  data: SpectrogramData;
  viewParams: ViewParams;
  setViewParams: (viewParams: ViewParams) => void;
  plotProps: PlotProps;
};

function AmplitudeSlider({
  data,
  viewParams,
  setViewParams,
  plotProps,
}: AmplitudeSliderInfo) {
  const onAmplitudeRangeChange = async ({
    start,
    end,
  }: {
    start: number;
    end: number;
  }) => {
    const params = SpectrogramViewParamsSchema.parse(viewParams);
    params.amplitude_min = Math.pow(10, start);
    params.amplitude_max = Math.pow(10, end);
    setViewParams(params);
  };

  const numDigits = plotProps.numSignificantDigits || 4;
  const smallPrecisionFactor = Math.pow(10, -1 * numDigits);
  const largePrecisionFactor = Math.pow(10, numDigits);

  let ampValues = data.amplitude.flat();
  ampValues = ampValues.map((x: number) =>
    Math.log10(Math.max(x, smallPrecisionFactor)),
  );

  const displayAmplitudeValues = (val: number) => {
    return `${Math.round(Math.pow(10, val) * largePrecisionFactor) / largePrecisionFactor}`;
  };

  return (
    <DataRangeSlider
      name={"Amplitude Range"}
      data={ampValues}
      onChange={onAmplitudeRangeChange}
      getValueLabel={(val) =>
        `${displayAmplitudeValues(val.start)} - ${displayAmplitudeValues(val.end)}`
      }
    />
  );
}

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
    setAnnotations,
    viewParams,
    setViewParams,
    plotProps,
    setPlotProps,
    isValidated,
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
      component: <ShotLabels labels={labels}></ShotLabels>,
    });

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
  } else if (data && project.task == TaskType.Spectrogram) {
    const resultSpec = SpectrogramDataSchema.safeParse(data);
    if (!resultSpec.success) {
      console.warn("MHD spectrogram data is not available");
      return null;
    }

    const mhdData = resultSpec.data;
    tools.push({
      name: "Amplitude Range",
      component: (
        <AmplitudeSlider
          data={mhdData}
          viewParams={viewParams}
          setViewParams={setViewParams}
          plotProps={plotProps}
        />
      ),
    });

    tools.push({
      name: "Color Map",
      component: (
        <ColorMapPicker plotProps={plotProps} setPlotProps={setPlotProps} />
      ),
    });

    tools.push({
      name: "Threshold",
      component: (
        <SpectrogramThresholdTool
          project_id={project_id}
          sample_id={sample_id}
          signal_name={"mirnov"}
          plotProps={plotProps}
          setPlotProps={setPlotProps}
        />
      ),
    });
  } else if (data && project.task === TaskType.Video) {
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
