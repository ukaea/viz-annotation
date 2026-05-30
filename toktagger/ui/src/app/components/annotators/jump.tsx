import { useEffect, useState } from "react";
import {
  Provider,
  defaultTheme,
  Slider,
  Flex,
  ComboBox,
  Item,
  Switch,
} from "@adobe/react-spectrum";
import { Annotation, MultiVariateTimeSeriesData } from "@/types";
import { AnnotatorTypes } from "./types";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { useSample } from "@/app/contexts/SampleContext";

type JumpDetectionType = {
  project_id: string;
  sample_id: string;
  data: MultiVariateTimeSeriesData;
};

export function JumpDetectionTool({
  project_id,
  sample_id,
  data,
}: JumpDetectionType) {
  const { annotations, dataParams, setAnnotations } = useSample();

  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) => ann.created_by === AnnotatorTypes.JUMP_DETECTION,
    );
  });

  const [signalName, setSignalName] = useState<string | null>(null);
  const signalOptions = Object.keys(data.values).map((value, index) => ({
    id: index,
    name: value,
  }));
  const [threshold, setThreshold] = useState<number>(2);
  const [minDistance, setMinDistance] = useState<number>(5);
  const [smoothingValue, setSmoothingValue] = useState<number>(2);
  const [numPoints, setNumPoints] = useState<number>(2000);
  const validSignalName = signalName && signalName in data.values;

  useEffect(() => {
    const fetchData = async () => {
      if (!isEnabled) {
        // Remove previous annotations from this annotator
        setAnnotations((previousAnnotations: Annotation[]) => {
          const otherAnnotations = previousAnnotations.filter(
            (annotation: Annotation) =>
              annotation.created_by !== AnnotatorTypes.JUMP_DETECTION ||
              annotation.validated,
          );
          return otherAnnotations;
        });
        return;
      } else if (!validSignalName) {
        return;
      }

      const response = await apiFetch(
        `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotator/jump_detection`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            annotator_params: {
              signal_name: signalName,
              threshold: threshold,
              min_distance: minDistance,
              smoothing: smoothingValue,
              num_points: numPoints,
            },
            data_params: dataParams,
          }),
        },
      );

      const payload: Annotation[] = await response.json();
      setAnnotations((previousAnnotations: Annotation[]) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !== AnnotatorTypes.JUMP_DETECTION ||
            annotation.validated,
        );
        return otherAnnotations.concat(payload);
      });
    };
    fetchData();
  }, [
    project_id,
    sample_id,
    signalName,
    minDistance,
    threshold,
    smoothingValue,
    numPoints,
    isEnabled,
    validSignalName,
    dataParams,
    setAnnotations,
  ]);

  return (
    <Provider theme={defaultTheme}>
      <div className="m-4">
        <Flex direction="column">
          <Switch isSelected={isEnabled} onChange={setIsEnabled}>
            Enable Tool
          </Switch>
          <ComboBox
            label="Signal Name"
            defaultItems={signalOptions}
            onInputChange={setSignalName}
          >
            {(x) => <Item>{x.name}</Item>}
          </ComboBox>
          <br />
          <Slider
            label="Threshold"
            minValue={0.5}
            maxValue={5}
            step={0.01}
            defaultValue={threshold}
            onChangeEnd={setThreshold}
          />
          <br />
          <Slider
            label="Minimum Distance"
            minValue={1}
            maxValue={100}
            step={1}
            defaultValue={minDistance}
            onChangeEnd={setMinDistance}
          />
          <br />
          <Slider
            label="Smoothing"
            minValue={0.1}
            maxValue={5}
            step={0.1}
            defaultValue={smoothingValue}
            onChangeEnd={setSmoothingValue}
          />
          <br />
          <Slider
            label="No. Points"
            minValue={100}
            maxValue={5000}
            step={10}
            defaultValue={numPoints}
            onChangeEnd={setNumPoints}
          />
        </Flex>
      </div>
    </Provider>
  );
}
