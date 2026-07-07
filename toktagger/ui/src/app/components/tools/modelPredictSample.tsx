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
import { Annotations, Annotation } from "@/types";
import {
  getModelTypes,
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
  const [isEnabled, setIsEnabled] = useState<boolean>(() => {
    return annotations.some(
      (ann) => project?.model_types.includes(ann.created_by) || false,
    );
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [modelNames, setModelNames] = useState<string[] | null>(null);
  const [selectedModelName, setSelectedModelName] = useState<string | null>(
    null,
  );
  const [useGPU, setUseGPU] = useState<boolean>(false);
  const [schema, setSchema] = useState<RJSFSchema | null>(null);
  const [unvalidatedFormData, setUnvalidatedFormData] = useState<
    Record<string, unknown>
  >({});
  const formRef = useRef<Form>(null);

  useEffect(() => {
    if (!isEnabled || !project) {
      return;
    }
    (async () => {
      const response = await getModelTypes(project.task);

      if (response.ok) {
        const data = await response.json();
        const modelTypes = data as string[];
        setModelNames(modelTypes);
      } else {
        const errorMessage = await response.json();
        setMessage(errorMessage.detail);
      }
    })();
  }, [isEnabled, project]);

  useEffect(() => {
    const updateSchema = async () => {
      if (!selectedModelName) {
        setSchema(null);
        return;
      }
      const newSchema: RJSFSchema | null =
        await getModelPredictSchema(selectedModelName);
      setSchema(newSchema);
    };
    updateSchema();
  }, [selectedModelName]);

  const onEnable = (newIsEnabled: boolean) => {
    setIsEnabled(newIsEnabled);
    if (!newIsEnabled) {
      // Remove previous annotations from this model
      setAnnotations((previousAnnotations: Annotations) => {
        const otherAnnotations = previousAnnotations.filter(
          (annotation: Annotation) =>
            annotation.created_by !== selectedModelName || annotation.validated,
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
    if (!selectedModelName || !project) {
      return;
    }

    const response = await startSamplePredictions(
      project_id,
      sample_id,
      selectedModelName,
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
    if (!taskId || !selectedModelName || !isEnabled) return;

    let pollCounter = 0;
    // Poll for result from GET predictions endpoint
    const interval = setInterval(async () => {
      if (selectedModelName == null) {
        clearInterval(interval);
        setIsLoading(false);
        return;
      }
      const response = await getSamplePredictions(
        project_id,
        sample_id,
        selectedModelName,
        taskId,
      );
      const payload = await response.json();

      if (response.status === 202) {
        // Predictions queued but not done yet, so continue to poll
        pollCounter += 1;
        if (pollCounter > 30) {
          setMessage("Predictions timed out - try refreshing the page later!");
          clearInterval(interval);
          setIsLoading(false);
        }
      } else if (response.ok) {
        setAnnotations((previousAnnotations: Annotations) => {
          return previousAnnotations.concat(payload);
        });
        clearInterval(interval);
        setIsLoading(false);
        setMessage(null);
      } else {
        setMessage(payload.detail);
        clearInterval(interval);
        setIsLoading(false);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [
    project_id,
    sample_id,
    selectedModelName,
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
            label="Select Model Type"
            validationState={message ? "invalid" : undefined}
            errorMessage={message}
            isDisabled={!isEnabled}
            selectedKey={selectedModelName}
            onSelectionChange={(key) =>
              setSelectedModelName(key !== null ? String(key) : null)
            }
          >
            {modelNames
              ? modelNames.map((model_name) => (
                  <Item key={model_name}>{model_name}</Item>
                ))
              : null}
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
              isDisabled={!isEnabled || !selectedModelName}
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
