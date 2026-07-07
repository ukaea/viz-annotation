import { useSample } from "@/app/contexts/SampleContext";
import { BACKEND_API_URL } from "@/app/core";
import { Annotation, Profile2DDataSchema, Profile2DViewParams } from "@/types";
import { Flex, NumberField, RangeSlider, Switch } from "@adobe/react-spectrum";
import { useEffect, useState } from "react";
import { AnnotatorTypes } from "./types";
import { z } from "zod";
import NumberStepper from "../ui/number_stepper";
import { getSignalNames } from "@/app/utils";

const Profile2DThresholdParamsSchema = z.object({
  signal_name: z.string(),
  percentile: z.number(),
  freq_max: z.number().default(50),
  freq_min: z.number().default(3),
  sigma: z.number().default(2),
  min_size: z.number().int().default(150),
  line_filter_width: z.number().int().default(0),
});

type Profile2DThresholdParams = z.infer<typeof Profile2DThresholdParamsSchema>;

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

  const params: Profile2DThresholdParams = Profile2DThresholdParamsSchema.parse(
    {
      signal_name: signalName,
      percentile: 95,
    },
  );

  // Tooling properties
  const [range, setRange] = useState<{
    start: number;
    end: number;
  }>({
    start: params.freq_min,
    end: params.freq_max,
  });

  const [defaultRange, setDefaultRange] = useState<{
    start: number;
    end: number;
  }>({
    start: params.freq_min,
    end: params.freq_max,
  });

  const [percentile, setPercentile] = useState(params.percentile);
  const [sigma, setSigma] = useState<number>(params.sigma);
  const [minSize, setMinSize] = useState<number>(params.min_size);
  const [lineFilterWidth, setLineFilterWidth] = useState(
    params.line_filter_width,
  );

  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) => ann.created_by === AnnotatorTypes.PROFILE2D_THRESHOLD,
    );
  });

  const onThresholdChange = (value: boolean) => {
    setIsEnabled(value);
    setPlotProps({ ...plotProps, thresholdActive: value });
  };

  useEffect(() => {
    if (!isEnabled) {
      return;
    }

    const fetchData = async () => {
      if (!signalName) return;

      if (!isEnabled) {
        // Remove previous annotations from this annotator
        setAnnotations((previousAnnotations: Annotation[]) => {
          const otherAnnotations = previousAnnotations.filter(
            (annotation: Annotation) =>
              annotation.created_by !== AnnotatorTypes.PROFILE2D_THRESHOLD ||
              annotation.validated,
          );
          return otherAnnotations;
        });
        return;
      }

      const response = await fetch(
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
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.signal_name !== signalName &&
            (annotation.created_by !== AnnotatorTypes.PROFILE2D_THRESHOLD ||
              annotation.validated),
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
    viewParams,
  ]);

  useEffect(() => {
    if (!isEnabled) return;
    if (!data || !Profile2DDataSchema.safeParse(data).success) return;

    const specData = Profile2DDataSchema.parse(data);

    const dim_1_MinValue = specData.dim_1[0];
    const dim_1_MaxValue = specData.dim_1[specData.dim_1.length - 1];
    setDefaultRange({
      start: dim_1_MinValue,
      end: dim_1_MaxValue,
    });
    setRange({
      start: dim_1_MinValue,
      end: dim_1_MaxValue,
    });
  }, [data, isEnabled]);

  return (
    <Flex direction="column" gap="size-200" justifyContent="start">
      <Switch isSelected={isEnabled} onChange={onThresholdChange}>
        Thresholding
      </Switch>
      {isEnabled && (
        <>
          <NumberStepper
            label="Percentile"
            defaultValue={percentile}
            onChange={(value: number) => {
              setPercentile(value);
            }}
          />
          <RangeSlider
            label="Range of Interest"
            value={range}
            minValue={defaultRange.start}
            maxValue={defaultRange.end}
            step={1}
            onChangeEnd={setRange}
          />
          <NumberField
            label="Sigma"
            value={sigma}
            minValue={0}
            onChange={setSigma}
            step={0.001}
          />
          <NumberField
            label="Min Size"
            value={minSize}
            minValue={0}
            onChange={setMinSize}
            formatOptions={{ maximumFractionDigits: 0 }}
          />
          <NumberField
            label="Vertical Line Filter Width"
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
