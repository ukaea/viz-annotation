import { useState, useEffect, useRef } from "react";
import {
  Provider,
  defaultTheme,
  ComboBox,
  Item,
  Flex,
  ProgressCircle,
  Switch,
  Button,
} from "@adobe/react-spectrum";
import { z } from "zod/v4";
import { Annotations, Annotation, Model, ModelSchema } from "@/types";
import {
  getModels,
  getModelPredictSchema,
  startSamplePredictions,
  getSamplePredictions,
} from "@/app/core";
import { useSample } from "@/app/contexts/SampleContext";
import ModelForm from "@/app/components/ui/schemaForm";
import { RJSFSchema } from "@rjsf/utils";
import Form from "@rjsf/core";

type ModelPredictInfo = {
  project_id: string;
  sample_id: string;
};

export function ModelPredictTool({ project_id, sample_id }: ModelPredictInfo) {
  const { annotations, project, dataParams, setAnnotations } = useSample();
  const [isEnabled, setIsEnabled] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [useGPU, setUseGPU] = useState<boolean>(false);
  const [schema, setSchema] = useState<RJSFSchema | null>(null);
  const [unvalidatedFormData, setUnvalidatedFormData] = useState<
    Record<string, unknown>
  >({});
  const formRef = useRef<Form>(null);
  const didAutoEnable = useRef<boolean>(false);

  const selectedModel =
    models.find((model) => model._id === selectedModelId) ?? null;
  const selectedModelType = selectedModel?.type ?? null;
  // Predictions are recorded against the model name, so match annotations on the
  // same value the backend stamps them with.
  const annotatorName = selectedModel
    ? (selectedModel.name ?? selectedModel.type)
    : null;

  // Refetch when the tool is switched on, so a model trained from this page
  // without a reload still shows up in the list.
  useEffect(() => {
    (async () => {
      const response = await getModels(project_id);

      if (!response.ok) {
        const errorMessage = await response.json();
        setMessage(errorMessage.detail);
        return;
      }

      const result = z.array(ModelSchema).safeParse(await response.json());
      if (!result.success) {
        setMessage("Could not read the models for this project!");
        return;
      }

      // Only a model which finished training can make predictions.
      setModels(
        result.data.filter((model) => model.training_status === "completed"),
      );
    })();
  }, [project_id, isEnabled]);

  // Start out enabled if this sample already has predictions from one of the
  // project's models, but only take over the switch before the user touches it.
  useEffect(() => {
    if (didAutoEnable.current || models.length === 0) {
      return;
    }
    const names = new Set(models.map((model) => model.name ?? model.type));
    if (annotations.some((annotation) => names.has(annotation.created_by))) {
      didAutoEnable.current = true;
      setIsEnabled(true);
    }
  }, [models, annotations]);

  useEffect(() => {
    const updateSchema = async () => {
      if (!selectedModelType) {
        setSchema(null);
        return;
      }
      const newSchema: RJSFSchema | null =
        await getModelPredictSchema(selectedModelType);
      setSchema(newSchema);
    };
    updateSchema();
  }, [selectedModelType]);

  const onEnable = (newIsEnabled: boolean) => {
    didAutoEnable.current = true;
    setIsEnabled(newIsEnabled);
    if (!newIsEnabled) {
      // Remove previous annotations from this model
      setAnnotations((previousAnnotations: Annotations) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !== annotatorName || annotation.validated,
        );
        return otherAnnotations;
      });
    }
    return;
  };

  const pressSubmit = () => {
    if (schema) {
      formRef.current?.submit();
    } else {
      submitPredictJob({});
    }
  };

  const submitPredictJob = async (params: Record<string, unknown>) => {
    if (!selectedModel || !project) {
      return;
    }

    const response = await startSamplePredictions(
      project_id,
      sample_id,
      selectedModel.type,
      selectedModel.version,
      useGPU,
      params,
      dataParams,
    );
    const payload = await response.json();

    if (response.ok) {
      setIsLoading(true);
      setTaskId(payload.task_id);
      setMessage(null);
    } else {
      setMessage(payload.detail);
    }
  };

  useEffect(() => {
    if (!taskId || !selectedModelType || !isEnabled) return;

    let pollCounter = 0;
    // Poll for result from GET predictions endpoint every 3 seconds.
    // DTW inference can take several seconds so polling faster just spams the server.
    const interval = setInterval(async () => {
      const response = await getSamplePredictions(
        project_id,
        sample_id,
        selectedModelType,
        taskId,
      );
      const payload = await response.json();

      if (response.status === 202) {
        // Predictions queued but not done yet, so continue to poll
        pollCounter += 1;
        if (pollCounter > 20) {
          setMessage("Predictions timed out - try refreshing the page later!");
          clearInterval(interval);
          setIsLoading(false);
        }
      } else if (response.ok) {
        setAnnotations((previousAnnotations: Annotations) => {
          // Replace any unvalidated predictions from this model with the new
          // results rather than appending, so repeated runs don't stack up.
          const withoutStale = previousAnnotations.filter(
            (ann: Annotation) =>
              ann.created_by !== annotatorName || ann.validated,
          );
          return [...withoutStale, ...payload];
        });
        clearInterval(interval);
        setIsLoading(false);
        setMessage(null);
      } else {
        setMessage(payload.detail);
        clearInterval(interval);
        setIsLoading(false);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [
    project_id,
    sample_id,
    selectedModelType,
    annotatorName,
    taskId,
    setAnnotations,
    isEnabled,
  ]);

  if (!project) {
    return;
  }

  return (
    <Provider theme={defaultTheme}>
      <div className="m-4">
        <Flex direction="column">
          <Switch isSelected={isEnabled} onChange={onEnable}>
            Enable Tool
          </Switch>
          <ComboBox
            label="Select Model"
            validationState={message ? "invalid" : undefined}
            errorMessage={message}
            description={
              models.length === 0
                ? "No trained models yet - train one for this project first."
                : undefined
            }
            isDisabled={!isEnabled || models.length === 0}
            selectedKey={selectedModelId}
            onSelectionChange={(key) => {
              setSelectedModelId(key !== null ? String(key) : null);
              setTaskId(null);
              setMessage(null);
              setIsLoading(false);
            }}
          >
            {models.map((model) => (
              <Item key={model._id}>{model.name ?? model.type}</Item>
            ))}
          </ComboBox>
          <ModelForm
            ref={formRef}
            schema={schema}
            onSubmit={submitPredictJob}
            disabled={!isEnabled}
            formData={unvalidatedFormData}
            setFormData={setUnvalidatedFormData}
            useGPU={useGPU}
            setUseGPU={setUseGPU}
          />
          <Flex marginTop="size-200" marginBottom="size-200">
            <Button
              marginEnd="size-400"
              variant="accent"
              isDisabled={!isEnabled || !selectedModel}
              onPress={pressSubmit}
            >
              Predict
            </Button>
            {isLoading ? (
              <ProgressCircle aria-label="Loading…" isIndeterminate />
            ) : null}
          </Flex>
        </Flex>
      </div>
    </Provider>
  );
}
