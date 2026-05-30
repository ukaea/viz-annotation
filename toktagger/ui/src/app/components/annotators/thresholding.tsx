import { useSample } from "@/app/contexts/SampleContext";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { Annotation, PlotProps } from "@/types";
import { ActionButton, Flex, NumberField, Switch } from "@adobe/react-spectrum";
import { useEffect, useState } from "react";
import { AnnotatorTypes } from "./types";

type SpectrogramThresholdToolInfo = {
  project_id: string;
  sample_id: string;
  signal_name: string;
  plotProps: PlotProps;
  setPlotProps: (props: PlotProps) => void;
};

export default function SpectrogramThresholdTool({
  project_id,
  sample_id,
  signal_name,
  plotProps,
  setPlotProps,
}: SpectrogramThresholdToolInfo) {
  const { annotations, setAnnotations } = useSample();
  const [value, setValue] = useState(95);

  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) => ann.created_by === AnnotatorTypes.SPECTROGRAM_THRESHOLD,
    );
  });

  const onThresholdChange = (value: boolean) => {
    setIsEnabled(value);
    setPlotProps({ ...plotProps, thresholdActive: value });
  };

  const incrementValue = (increment: number) => {
    setValue((prevValue) => {
      const newValue = prevValue + increment;
      if (newValue < 0) return 0;
      if (newValue > 99) return 99;
      return newValue;
    });
  };

  useEffect(() => {
    const fetchData = async () => {
      if (!isEnabled) {
        // Remove previous annotations from this annotator
        setAnnotations((previousAnnotations: Annotation[]) => {
          const otherAnnotations = previousAnnotations.filter(
            (annotation: Annotation) =>
              annotation.created_by !== AnnotatorTypes.SPECTROGRAM_THRESHOLD ||
              annotation.validated,
          );
          return otherAnnotations;
        });
        return;
      }

      const response = await apiFetch(
        `${BACKEND_API_URL}/projects/${project_id}/samples/${sample_id}/annotator/spectrogram_threshold`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            signal_name: signal_name,
            percentile: value,
          }),
        },
      );

      const payload: Annotation[] = await response.json();
      setAnnotations((previousAnnotations: Annotation[]) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !== AnnotatorTypes.SPECTROGRAM_THRESHOLD ||
            annotation.validated,
        );
        return otherAnnotations.concat(payload);
      });
    };

    fetchData();
  }, [project_id, sample_id, isEnabled, value, signal_name, setAnnotations]);

  return (
    <>
      <Switch isSelected={isEnabled} onChange={onThresholdChange}>
        Thresholding
      </Switch>
      {isEnabled && (
        <Flex
          direction="column"
          gap="size-100"
          margin={"size-200"}
          alignItems={"center"}
        >
          <NumberField
            label="Percentile"
            value={value}
            onChange={setValue}
            minValue={0}
            maxValue={99}
            hideStepper={true}
          />
          <Flex direction="row" gap="size-100">
            <ActionButton
              onPress={() => {
                incrementValue(-5);
              }}
            >
              -5
            </ActionButton>
            <ActionButton
              onPress={() => {
                incrementValue(-1);
              }}
            >
              -1
            </ActionButton>
            <ActionButton
              onPress={() => {
                incrementValue(1);
              }}
            >
              +1
            </ActionButton>
            <ActionButton
              onPress={() => {
                incrementValue(5);
              }}
            >
              +5
            </ActionButton>
          </Flex>
        </Flex>
      )}
    </>
  );
}
