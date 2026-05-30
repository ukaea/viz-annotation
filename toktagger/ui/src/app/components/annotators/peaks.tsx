"use client";
import { useEffect, useState } from "react";
import { Annotation, MultiVariateTimeSeriesData } from "@/types";
import {
  Provider,
  defaultTheme,
  Slider,
  Flex,
  ComboBox,
  Item,
  RangeSlider,
  Switch,
} from "@adobe/react-spectrum";
import { AnnotatorTypes } from "./types";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { useSample } from "@/app/contexts/SampleContext";

type PeakDetectionType = {
  project_id: string;
  sample_id: string;
  data: MultiVariateTimeSeriesData;
};
export function PeakDetectionTool({
  project_id,
  sample_id,
}: PeakDetectionType) {
  const { annotations, setAnnotations, dataParams, data } = useSample();

  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) => ann.created_by === AnnotatorTypes.PEAK_DETECTION,
    );
  });

  const [prominence, setProminance] = useState<number>(5);
  const [distance, setDistance] = useState<number>(1);

  const [timeMinDefault, setTimeMinDefault] = useState<number>(0);
  const [timeMaxDefault, setTimeMaxDefault] = useState<number>(0);
  const [timeRange, setTimeRange] = useState<{ start: number; end: number }>({
    start: 0,
    end: 100,
  });
  const [signalName, setSignalName] = useState<string | null>(null);
  const signalOptions = Object.keys(data.values).map((value, index) => ({
    id: index,
    name: value,
  }));

  const validSignal = signalName !== null && signalName in data.values;

  useEffect(() => {
    if (data && signalName !== null && signalName in data.values) {
      const time = data.values[signalName].time;
      const tmin = Math.min(...time);
      const tmax = Math.max(...time);
      setTimeMinDefault(tmin);
      setTimeMaxDefault(tmax);
    }
  }, [data, signalName]);

  useEffect(() => {
    const fetchData = async () => {
      if (!isEnabled) {
        // Remove previous annotations from this annotator
        setAnnotations((previousAnnotations: Annotation[]) => {
          const otherAnnotations = previousAnnotations.filter(
            (annotation: Annotation) =>
              annotation.created_by !== AnnotatorTypes.PEAK_DETECTION ||
              annotation.validated,
          );
          return otherAnnotations;
        });
        return;
      } else if (!validSignal) {
        return;
      }

      const response = await apiFetch(
        `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotator/peak_detection`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            annotator_params: {
              signal_name: signalName,
              prominence: prominence,
              distance: distance,
              time_min: timeRange.start,
              time_max: timeRange.end,
            },
            data_params: dataParams,
          }),
        },
      );

      const payload: Annotation[] = await response.json();
      setAnnotations((previousAnnotations: Annotation[]) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !== AnnotatorTypes.PEAK_DETECTION ||
            annotation.validated,
        );
        return otherAnnotations.concat(payload);
      });
    };

    fetchData();
  }, [
    project_id,
    sample_id,
    prominence,
    distance,
    timeRange,
    isEnabled,
    signalName,
    validSignal,
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
            defaultInputValue={signalName ?? undefined}
            defaultItems={signalOptions}
            onInputChange={setSignalName}
          >
            {(x) => <Item>{x.name}</Item>}
          </ComboBox>
          <br />
          <Slider
            label="Prominence"
            minValue={0.01}
            maxValue={10}
            defaultValue={prominence}
            step={0.001}
            onChangeEnd={setProminance}
          />
          <Slider
            label="Distance"
            minValue={1}
            maxValue={1000}
            defaultValue={distance}
            onChangeEnd={setDistance}
          />
          <RangeSlider
            label="Time Range"
            defaultValue={{ start: timeMinDefault, end: timeMaxDefault }}
            value={timeRange}
            onChangeEnd={setTimeRange}
            step={0.001}
            minValue={timeMinDefault}
            maxValue={timeMaxDefault}
          />
        </Flex>
      </div>
    </Provider>
  );
}
