import { useSample } from "@/app/contexts/SampleContext";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { Annotation, Profile2DDataSchema, Profile2DViewParams } from "@/types";
import {
  ActionButton,
  Flex,
  NumberField,
  RangeSlider,
  Switch,
} from "@adobe/react-spectrum";
import { useEffect, useState } from "react";
import { AnnotatorTypes, annotatorCreatedBy } from "./types";
import { getSignalNames } from "@/app/utils";

// Defaults for the thresholding parameters. The range of interest is replaced
// with the profile's own extent once data has loaded.
const DEFAULT_PERCENTILE = 95;
const DEFAULT_SIGMA = 2;
const DEFAULT_MIN_SIZE = 150;
const DEFAULT_LINE_FILTER_WIDTH = 0;

type Profile2DThresholdToolInfo = {
  project_id: string;
  sample_id: string;
};

export default function Profile2DThresholdTool({
  project_id,
  sample_id,
}: Profile2DThresholdToolInfo) {
  const {
    sample,
    annotations,
    setAnnotations,
    dataParams,
    data,
    viewParams,
    plotProps,
    setPlotProps,
  } = useSample();

  const signalNames = getSignalNames(sample);
  const signalName =
    (viewParams as Profile2DViewParams)?.signal_name || signalNames[0];

  const [percentile, setPercentile] = useState(DEFAULT_PERCENTILE);
  const [sigma, setSigma] = useState<number>(DEFAULT_SIGMA);
  const [minSize, setMinSize] = useState<number>(DEFAULT_MIN_SIZE);
  const [lineFilterWidth, setLineFilterWidth] = useState(
    DEFAULT_LINE_FILTER_WIDTH,
  );
  // `bounds` is the full extent of the profile and fixes the slider's limits;
  // `range` is the user's current selection within it.
  const [bounds, setBounds] = useState<{ start: number; end: number } | null>(
    null,
  );
  const [range, setRange] = useState<{ start: number; end: number } | null>(
    null,
  );

  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) =>
        ann.created_by === annotatorCreatedBy(AnnotatorTypes.PROFILE_2D_THRESHOLD),
    );
  });

  const onThresholdChange = (value: boolean) => {
    setIsEnabled(value);
    setPlotProps({ ...plotProps, thresholdActive: value });
  };

  const incrementValue = (increment: number) => {
    setPercentile((prevValue) => {
      const newValue = prevValue + increment;
      if (newValue < 0) return 0;
      if (newValue > 99) return 99;
      return newValue;
    });
  };

  // The range of interest defaults to the full extent of the loaded profile.
  useEffect(() => {
    if (!isEnabled || !data) return;

    const result = Profile2DDataSchema.safeParse(data);
    if (!result.success) return;

    const profile = result.data;
    const extent = {
      start: profile.dim_1[0],
      end: profile.dim_1[profile.dim_1.length - 1],
    };
    setBounds(extent);
    setRange(extent);
  }, [data, isEnabled]);

  useEffect(() => {
    if (!isEnabled) {
      // Drop this annotator's unsaved output so it doesn't linger after toggling off.
      setAnnotations((previousAnnotations: Annotation[]) =>
        previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !==
              annotatorCreatedBy(AnnotatorTypes.PROFILE_2D_THRESHOLD) ||
            annotation.validated,
        ),
      );
      return;
    }

    const fetchData = async () => {
      if (!signalName || !range) return;

      const response = await apiFetch(
        `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotator/profile_2d_threshold`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            annotator_params: {
              signal_name: signalName,
              percentile: percentile,
              dim_1_min: range.start,
              dim_1_max: range.end,
              sigma: isNaN(sigma) ? 0 : sigma,
              min_size: isNaN(minSize) ? 0 : minSize,
              line_filter_width: lineFilterWidth,
            },
            data_params: dataParams,
          }),
        },
      );

      const payload: Annotation[] = await response.json();
      setAnnotations((previousAnnotations: Annotation[]) => {
        // Replace only this annotator's unsaved output for the current signal.
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            !(
              annotation.created_by ===
                annotatorCreatedBy(AnnotatorTypes.PROFILE_2D_THRESHOLD) &&
              !annotation.validated &&
              annotation.signal_name === signalName
            ),
        );
        return otherAnnotations.concat(payload);
      });
    };

    fetchData();
  }, [
    project_id,
    sample_id,
    isEnabled,
    percentile,
    signalName,
    setAnnotations,
    dataParams,
    range,
    lineFilterWidth,
    minSize,
    sigma,
  ]);

  return (
    <Flex
      direction="column"
      gap="size-200"
      alignItems="start"
      justifyContent="start"
    >
      <Switch isSelected={isEnabled} onChange={onThresholdChange}>
        Thresholding
      </Switch>
      {isEnabled && (
        <>
          <NumberField
            label="Percentile"
            width="100%"
            value={percentile}
            onChange={setPercentile}
            minValue={0}
            maxValue={99}
            hideStepper={true}
          />
          <Flex direction="row" gap="size-100">
            <ActionButton onPress={() => incrementValue(-5)}>-5</ActionButton>
            <ActionButton onPress={() => incrementValue(-1)}>-1</ActionButton>
            <ActionButton onPress={() => incrementValue(1)}>+1</ActionButton>
            <ActionButton onPress={() => incrementValue(5)}>+5</ActionButton>
          </Flex>
          {bounds && range && (
            <RangeSlider
              label="Range of Interest"
              width="100%"
              value={range}
              minValue={bounds.start}
              maxValue={bounds.end}
              step={1}
              onChangeEnd={setRange}
            />
          )}
          <NumberField
            label="Sigma"
            width="100%"
            value={sigma}
            minValue={0}
            onChange={setSigma}
            step={0.001}
          />
          <NumberField
            label="Min Size"
            width="100%"
            value={minSize}
            minValue={0}
            onChange={setMinSize}
            formatOptions={{ maximumFractionDigits: 0 }}
          />
          <NumberField
            label="Vertical Line Filter Width"
            width="100%"
            value={lineFilterWidth}
            minValue={0}
            onChange={setLineFilterWidth}
            formatOptions={{ maximumFractionDigits: 0 }}
          />
        </>
      )}
    </Flex>
  );
}
