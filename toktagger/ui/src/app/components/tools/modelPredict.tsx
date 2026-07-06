import { useState, useEffect, useRef } from "react";
import {
  NumberField,
  TableView,
  Cell,
  Column,
  Row,
  TableBody,
  TableHeader,
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
  Selection,
  Tooltip,
  TooltipTrigger,
} from "@adobe/react-spectrum";
import Workflow from "@spectrum-icons/workflow/Workflow";
import CheckmarkCircle from "@spectrum-icons/workflow/CheckmarkCircle";
import Alert from "@spectrum-icons/workflow/Alert";
import { Project, Model } from "@/types";
import {
  startPredictions,
  getModels,
  stopTraining,
  getModelPredictSchema,
} from "@/app/core";
import ModelForm from "@/app/components/ui/schemaForm";

import { RJSFSchema } from "@rjsf/utils";
import Form from "@rjsf/core";

export function ModelPredictModal({
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
  const [models, setModels] = useState<Model[] | null>(null);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [useGPU, setUseGPU] = useState<boolean>(false);
  const [schema, setSchema] = useState<RJSFSchema | null>(null);
  const [unvalidatedFormData, setUnvalidatedFormData] = useState<
    Record<string, unknown>
  >({});
  const formRef = useRef<Form>(null);
  const [selectedKeys, setSelectedKeys] = useState<Selection | undefined>(
    undefined,
  );
  const [numPredictions, setNumPredictions] = useState<number>(20);
  const onSelectModel = (keys: Selection) => {
    setSelectedKeys(keys);

    if (keys === "all") {
      // Won't happen in single mode
      return;
    }

    if (!keys || keys.size === 0) {
      setSelectedModel(null);
      setSchema(null);
      return;
    }

    // Single select mode, so only ever one key
    const [key] = keys as Set<string>;

    if (!models) {
      return;
    }

    const model = models.find((model) => model._id === key);

    if (!model) {
      setMessage("Selected model could not be found!");
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      return;
    }

    setSelectedModel(model);
    return;
  };

  useEffect(() => {
    const updateSchema = async () => {
      if (!selectedModel || selectedModel.training_status != "completed") {
        setSchema(null);
        return;
      }
      const newSchema: RJSFSchema | null = await getModelPredictSchema(
        selectedModel.type,
      );
      setSchema(newSchema);
    };
    updateSchema();
  }, [selectedModel]);

  useEffect(() => {
    const fetchModels = async () => {
      if (!project._id) return;
      const response = await getModels(project._id);

      if (response.ok) {
        const data = await response.json();
        const models = data as Model[];
        setModels(models);
      } else {
        const errorMessage = await response.json();
        setMessage(errorMessage.detail);
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      }
    };

    let poll: ReturnType<typeof setInterval>;
    if (modalOpen) {
      setMessage(null);
      setMessageIcon(null);
      setSelectedModel(null);
      fetchModels();

      poll = setInterval(() => {
        fetchModels();
      }, 5000);
    }
    return () => {
      if (poll) clearInterval(poll);
    };
  }, [project._id, modalOpen]);

  const pressSubmit = () => {
    if (schema) {
      formRef.current?.submit();
    } else {
      submitPredictJob({});
    }
  };

  const submitPredictJob = async (params: Record<string, unknown>) => {
    if (!project._id || !models || !selectedKeys || !selectedModel) {
      return;
    }

    const response = await startPredictions(
      project._id,
      selectedModel.type,
      selectedModel.version,
      numPredictions,
      useGPU,
      params,
    );

    if (response.ok) {
      setMessage("Model predictions added to job queue!");
      setMessageIcon(
        <CheckmarkCircle aria-label="Success" color="positive" size="S" />,
      );
      setSelectedKeys(undefined);
    } else {
      const errorMessage = await response.json();
      setMessage(errorMessage.detail);
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
    }
  };

  const stopTrainingJob = async () => {
    if (!project._id || !models || !selectedKeys || !selectedModel) {
      return;
    }

    const response = await stopTraining(
      project._id,
      selectedModel.type,
      selectedModel.version,
    );

    if (response.ok) {
      setMessage("Model training has been stopped!");
      setMessageIcon(
        <CheckmarkCircle aria-label="Success" color="positive" size="S" />,
      );
      setSelectedKeys(undefined);
    } else {
      const errorMessage = await response.json();
      setMessage(errorMessage.detail);
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
    }
  };

  return (
    <DialogTrigger onOpenChange={(isOpen) => setModalOpen(isOpen)}>
      <TooltipTrigger delay={350} placement="bottom">
        <ActionButton
          isDisabled={!isEnabled}
          aria-label="Create Predictions from ML Model"
        >
          <Workflow />
        </ActionButton>
        <Tooltip>"Make Predictions"</Tooltip>
      </TooltipTrigger>
      {(close) => (
        <Dialog>
          <Heading>
            <Flex alignItems="center" gap="size-100">
              <Workflow size="S" />
              <Text>Create Predictions from ML Model</Text>
            </Flex>
          </Heading>
          <Divider />
          <Content>
            <Flex
              justifyContent="space-between"
              alignItems="center"
              marginBottom="size-200"
            >
              <NumberField
                label="Number of Predictions"
                onChange={setNumPredictions}
                defaultValue={20}
                minValue={10}
                step={10}
              />
              <Button
                variant="negative"
                isDisabled={
                  !selectedKeys ||
                  !models ||
                  !selectedModel ||
                  !["started", "queued"].includes(selectedModel.training_status)
                }
                onPress={stopTrainingJob}
              >
                Cancel Training
              </Button>
            </Flex>
            {models && (
              <TableView
                flex
                selectionMode="single"
                selectedKeys={selectedKeys}
                onSelectionChange={onSelectModel}
                height="size-3000"
                aria-label="Model Prediction Table"
              >
                <TableHeader>
                  <Column>Model Type</Column>
                  <Column>Version</Column>
                  <Column>Status</Column>
                  <Column>Score</Column>
                </TableHeader>
                <TableBody items={models}>
                  {(item) => (
                    <Row key={item["_id"]}>
                      <Cell>{item["type"]}</Cell>
                      <Cell>{item["version"]}</Cell>
                      <Cell>
                        {item["training_status"] === "started"
                          ? "Training: " + Math.round(item["progress"]) + "%"
                          : item["training_status"]}
                      </Cell>
                      <Cell>{Math.round(item["score"])}</Cell>
                    </Row>
                  )}
                </TableBody>
              </TableView>
            )}
            <ModelForm
              ref={formRef}
              schema={schema}
              onSubmit={submitPredictJob}
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
              isDisabled={
                !selectedKeys ||
                !models ||
                !selectedModel ||
                selectedModel.training_status != "completed"
              }
              onPress={pressSubmit}
            >
              Predict
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}
