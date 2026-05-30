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

enum ChangePointMethod {
  PELT = "pelt",
  HMM = "hmm",
}

type ChangePointDetectionType = {
  project_id: string;
  sample_id: string;
  data: MultiVariateTimeSeriesData;
};

export function ChangePointDetectionTool({
  project_id,
  sample_id,
  data,
}: ChangePointDetectionType) {
  const { annotations, dataParams, setAnnotations } = useSample();

  const methodOptions = [
    { id: 0, name: ChangePointMethod.PELT },
    { id: 1, name: ChangePointMethod.HMM },
  ];
  const signalOptions = Object.keys(data.values).map((value, index) => ({
    id: index,
    name: value,
  }));

  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) => ann.created_by === AnnotatorTypes.CHANGE_POINT_DETECTION,
    );
  });

  const [signalName, setSignalName] = useState<string | null>(null);
  const [penalty, setPenalty] = useState<number>(5);
  const [numPoints, setNumPoints] = useState<number>(500);
  const [method, setMethod] = useState<string>(ChangePointMethod.PELT);
  const [numComponents, setNumComponents] = useState<number>(3);
  const validSignalName = signalName && signalName in data.values;

  useEffect(() => {
    const fetchData = async () => {
      if (!isEnabled) {
        // Remove previous annotations from this annotator
        setAnnotations((previousAnnotations: Annotation[]) => {
          const otherAnnotations = previousAnnotations.filter(
            (annotation: Annotation) =>
              annotation.created_by !== AnnotatorTypes.CHANGE_POINT_DETECTION ||
              annotation.validated,
          );
          return otherAnnotations;
        });
        return;
      } else if (!validSignalName) {
        return;
      }

      const response = await apiFetch(
        `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotator/change_point_detection`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            annotator_params: {
              signal_name: signalName,
              method: method,
              penalty: penalty,
              num_points: numPoints,
              num_components: numComponents,
            },
            data_params: dataParams,
          }),
        },
      );

      const payload: Annotation[] = await response.json();
      setAnnotations((previousAnnotations) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !== AnnotatorTypes.CHANGE_POINT_DETECTION ||
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
    penalty,
    method,
    numPoints,
    numComponents,
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
          <ComboBox
            label="Method"
            defaultItems={methodOptions}
            defaultInputValue={method}
            onInputChange={setMethod}
          >
            {(x) => <Item>{x.name}</Item>}
          </ComboBox>
          <br />
          {method === ChangePointMethod.PELT && (
            <>
              <Slider
                label="Penalty"
                minValue={0.01}
                maxValue={30}
                defaultValue={penalty}
                step={0.001}
                onChangeEnd={setPenalty}
              />
              <br />
            </>
          )}
          {method === ChangePointMethod.HMM && (
            <>
              <Slider
                label="No. Components"
                minValue={1}
                maxValue={10}
                defaultValue={numComponents}
                step={1}
                onChangeEnd={setNumComponents}
              />
              <br />
            </>
          )}
          <Slider
            label="No. Points"
            minValue={100}
            maxValue={1000}
            defaultValue={numPoints}
            step={10}
            onChangeEnd={setNumPoints}
          />
        </Flex>
      </div>
    </Provider>
  );
}
