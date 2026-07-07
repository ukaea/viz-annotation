import { useState, useEffect, useRef } from "react";
import {
  ComboBox,
  Item,
  Flex,
  ActionButton,
  Button,
  ButtonGroup,
  Content,
  Dialog,
  DialogTrigger,
  Divider,
  Footer,
  Heading,
  Text,
  Tooltip,
  TooltipTrigger,
} from "@adobe/react-spectrum";
import WorkflowAdd from "@spectrum-icons/workflow/WorkflowAdd";
import CheckmarkCircle from "@spectrum-icons/workflow/CheckmarkCircle";
import Alert from "@spectrum-icons/workflow/Alert";
import { Project } from "@/types";
import { startTraining, getModelTypes, getModelTrainSchema } from "@/app/core";
import ModelForm from "@/app/components/ui/schemaForm";
import { RJSFSchema } from "@rjsf/utils";
import Form from "@rjsf/core";

export function ModelTrainModal({
  project,
  isEnabled,
}: {
  project: Project;
  isEnabled: boolean;
}) {
  const [modalOpen, setModalOpen] = useState<boolean>(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageIcon, setMessageIcon] = useState<React.JSX.Element | null>(
    null,
  );
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
    if (!modalOpen) {
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
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      }
    })();
  }, [modalOpen, project.task]);

  useEffect(() => {
    const updateSchema = async () => {
      if (!selectedModelName) {
        setSchema(null);
        return;
      }
      const newSchema: RJSFSchema | null =
        await getModelTrainSchema(selectedModelName);
      setSchema(newSchema);
    };
    updateSchema();
  }, [selectedModelName]);

  const pressSubmit = () => {
    if (schema) {
      formRef.current?.submit();
    } else {
      submitTrainJob({});
    }
  };

  const submitTrainJob = async (params: Record<string, unknown>) => {
    if (!selectedModelName || !project._id) {
      return;
    }
    const response = await startTraining(
      project._id,
      selectedModelName,
      useGPU,
      params,
    );
    if (response.ok) {
      setMessage("Model training added to job queue!");
      setMessageIcon(
        <CheckmarkCircle aria-label="Success" color="positive" size="S" />,
      );
    } else {
      const errorMessage = await response.json();
      setMessage(errorMessage.detail);
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
    }
  };

  return (
    <DialogTrigger onOpenChange={(isOpen) => setModalOpen(isOpen)}>
      <TooltipTrigger delay={350} placement="bottom">
        <ActionButton isDisabled={!isEnabled} aria-label="Train ML Model">
          <WorkflowAdd />
        </ActionButton>
        <Tooltip>"Train Model"</Tooltip>
      </TooltipTrigger>
      {(close) => (
        <Dialog>
          <Heading>
            <Flex alignItems="center" gap="size-100">
              <WorkflowAdd size="S" />
              <Text>Train ML Model</Text>
            </Flex>
          </Heading>
          <Divider />
          <Content>
            <ComboBox
              label="Select Model Type"
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
              onSubmit={submitTrainJob}
              formData={unvalidatedFormData}
              setFormData={setUnvalidatedFormData}
              useGPU={useGPU}
              setUseGPU={setUseGPU}
            />
          </Content>
          <Footer>
            {message && (
              <Text>
                {messageIcon} {message}
              </Text>
            )}
          </Footer>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Close
            </Button>
            <Button
              variant="accent"
              onPress={pressSubmit}
              isDisabled={!modelNames || !selectedModelName}
            >
              Train
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}
