import { useState, useEffect, useRef, useMemo } from "react";
import type { SortDescriptor } from "@react-types/shared";
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
  ProgressCircle,
  TableView,
  TableHeader,
  TableBody,
  Column,
  Row,
  Cell,
  Tabs,
  TabList,
  TabPanels,
  Text,
  Tooltip,
  TooltipTrigger,
  Well,
} from "@adobe/react-spectrum";
import WorkflowAdd from "@spectrum-icons/workflow/WorkflowAdd";
import CheckmarkCircle from "@spectrum-icons/workflow/CheckmarkCircle";
import Alert from "@spectrum-icons/workflow/Alert";
import { Project, Model } from "@/types";
import {
  startTraining,
  getModelTypes,
  getModelTrainSchema,
  getModelMeta,
  getModels,
  stopTraining,
} from "@/app/core";
import ModelForm from "@/app/components/ui/schemaForm";
import { RJSFSchema } from "@rjsf/utils";
import Form from "@rjsf/core";

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleString(undefined, {
      dateStyle: "short",
      timeStyle: "short",
    });
  } catch {
    return ts;
  }
}

export function ModelTrainModal({
  project,
  isEnabled,
}: {
  project: Project;
  isEnabled: boolean;
}) {
  const [modalOpen, setModalOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>("train");
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
  const [modelDescription, setModelDescription] = useState<string | null>(null);
  const [trainingModelId, setTrainingModelId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);
  const [models, setModels] = useState<Model[] | null>(null);
  const [sortDescriptor, setSortDescriptor] = useState<SortDescriptor>({
    column: "timestamp",
    direction: "descending",
  });
  const formRef = useRef<Form>(null);

  const sortedModels = useMemo(() => {
    if (!models) return null;
    return [...models].sort((a, b) => {
      const col = String(sortDescriptor.column);
      let cmp = 0;
      if (col === "version" || col === "score") {
        cmp =
          (a[col as keyof Model] as number) - (b[col as keyof Model] as number);
      } else if (col === "timestamp") {
        cmp = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else {
        cmp = String(a[col as keyof Model] ?? "").localeCompare(
          String(b[col as keyof Model] ?? ""),
        );
      }
      return sortDescriptor.direction === "descending" ? -cmp : cmp;
    });
  }, [models, sortDescriptor]);

  // Fetch available model types when modal opens
  useEffect(() => {
    if (!modalOpen) return;

    (async () => {
      const response = await getModelTypes(project.task);
      if (response.ok) {
        const data = await response.json();
        setModelNames(data as string[]);
      } else {
        const errorMessage = await response.json();
        setMessage(errorMessage.detail);
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      }
    })();
  }, [modalOpen, project.task]);

  // Fetch train schema when model selection changes
  useEffect(() => {
    setSchema(null);
    if (!selectedModelName) return;
    (async () => {
      const newSchema: RJSFSchema | null =
        await getModelTrainSchema(selectedModelName);
      setSchema(newSchema);
    })();
  }, [selectedModelName]);

  // Fetch model description when model selection changes
  useEffect(() => {
    setModelDescription(null);
    if (!selectedModelName) return;
    (async () => {
      try {
        const meta = await getModelMeta(selectedModelName);
        setModelDescription(meta.description);
      } catch {
        // description is optional — silently ignore
      }
    })();
  }, [selectedModelName]);

  // Poll model list while modal is open (Trained Models tab + training spinner)
  useEffect(() => {
    if (!modalOpen || !project._id) return;

    const fetchModels = async () => {
      const response = await getModels(project._id!);
      if (!response.ok) {
        const errorMessage = await response.json();
        setMessage(errorMessage.detail);
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
        return;
      }
      const data = (await response.json()) as Model[];
      setModels(data);

      if (trainingModelId) {
        const m = data.find((x) => x._id === trainingModelId);
        if (m) {
          setTrainingStatus(m.training_status);
          if (m.training_status === "completed") {
            setMessage(`Training complete! Score: ${Math.round(m.score)}%`);
            setMessageIcon(
              <CheckmarkCircle
                aria-label="Success"
                color="positive"
                size="S"
              />,
            );
          } else if (m.training_status === "failed") {
            setMessage("Training failed.");
            setMessageIcon(
              <Alert aria-label="Failed" color="negative" size="S" />,
            );
          }
        }
      }
    };

    // Fetch immediately on open or tab switch (trainingModelId is null),
    // but not right after submitting a job — the "added to queue" message
    // should stay visible until the next regular interval tick.
    if (!trainingModelId) {
      fetchModels();
    }
    const poll = setInterval(fetchModels, 5000);
    return () => clearInterval(poll);
  }, [modalOpen, project._id, trainingModelId]);

  const handleModelSelection = (key: React.Key | null) => {
    setSelectedModelName(key !== null ? String(key) : null);
    setTrainingModelId(null);
    setTrainingStatus(null);
    setMessage(null);
    setMessageIcon(null);
    setUnvalidatedFormData({});
  };

  const handleOpenChange = (isOpen: boolean) => {
    setModalOpen(isOpen);
    if (!isOpen) {
      setActiveTab("train");
      setTrainingModelId(null);
      setTrainingStatus(null);
      setMessage(null);
      setMessageIcon(null);
    }
  };

  const pressSubmit = () => {
    if (schema) {
      formRef.current?.submit();
    } else {
      submitTrainJob({});
    }
  };

  const submitTrainJob = async (params: Record<string, unknown>) => {
    if (!selectedModelName || !project._id) return;

    const response = await startTraining(
      project._id,
      selectedModelName,
      useGPU,
      params,
    );

    if (response.ok) {
      const data = await response.json();
      setTrainingModelId(data.model_id);
      setTrainingStatus("queued");
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

  const stopTrainingJob = async (model: Model) => {
    if (!project._id) return;
    const response = await stopTraining(project._id, model.type, model.version);
    if (response.ok) {
      setMessage("Model training cancelled.");
      setMessageIcon(
        <CheckmarkCircle aria-label="Success" color="positive" size="S" />,
      );
    } else {
      const errorMessage = await response.json();
      setMessage(errorMessage.detail);
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
    }
  };

  const isTrainingActive =
    trainingStatus === "queued" || trainingStatus === "started";

  return (
    <DialogTrigger onOpenChange={handleOpenChange}>
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
              <Text>ML Models</Text>
            </Flex>
          </Heading>
          <Divider />
          <Content>
            <Tabs
              selectedKey={activeTab}
              onSelectionChange={(k) => setActiveTab(String(k))}
            >
              <TabList>
                <Item key="train">Train</Item>
                <Item key="models">Trained Models</Item>
              </TabList>
              <TabPanels>
                <Item key="train">
                  <Flex direction="column" gap="size-150" marginTop="size-150">
                    <ComboBox
                      label="Select Model Type"
                      selectedKey={selectedModelName}
                      onSelectionChange={handleModelSelection}
                    >
                      {modelNames
                        ? modelNames.map((name) => (
                            <Item key={name}>{name}</Item>
                          ))
                        : null}
                    </ComboBox>
                    {modelDescription && (
                      <Well>
                        <Text>{modelDescription}</Text>
                      </Well>
                    )}
                    {schema && (
                      <ModelForm
                        ref={formRef}
                        schema={schema}
                        onSubmit={submitTrainJob}
                        formData={unvalidatedFormData}
                        setFormData={setUnvalidatedFormData}
                        useGPU={useGPU}
                        setUseGPU={setUseGPU}
                      />
                    )}
                  </Flex>
                </Item>
                <Item key="models">
                  <Flex direction="column" gap="size-150" marginTop="size-150">
                    {sortedModels && sortedModels.length > 0 ? (
                      <TableView
                        aria-label="Trained models"
                        height="size-3000"
                        selectionMode="none"
                        sortDescriptor={sortDescriptor}
                        onSortChange={setSortDescriptor}
                      >
                        <TableHeader>
                          <Column key="type" allowsSorting>
                            Type
                          </Column>
                          <Column key="version" allowsSorting>
                            Version
                          </Column>
                          <Column key="timestamp" allowsSorting>
                            Created
                          </Column>
                          <Column key="training_status" allowsSorting>
                            Status
                          </Column>
                          <Column key="score" allowsSorting>
                            Score
                          </Column>
                          <Column key="actions">Actions</Column>
                        </TableHeader>
                        <TableBody items={sortedModels}>
                          {(item) => (
                            <Row key={item._id}>
                              <Cell>{item.type}</Cell>
                              <Cell>{item.version}</Cell>
                              <Cell>{formatTimestamp(item.timestamp)}</Cell>
                              <Cell>
                                {item.training_status === "started"
                                  ? "Training…"
                                  : item.training_status}
                              </Cell>
                              <Cell>
                                {item.training_status === "completed"
                                  ? Math.round(item.score)
                                  : "—"}
                              </Cell>
                              <Cell>
                                {["queued", "started"].includes(
                                  item.training_status,
                                ) ? (
                                  <Button
                                    variant="negative"
                                    onPress={() => stopTrainingJob(item)}
                                  >
                                    Cancel
                                  </Button>
                                ) : (
                                  <Text>—</Text>
                                )}
                              </Cell>
                            </Row>
                          )}
                        </TableBody>
                      </TableView>
                    ) : (
                      <Text>No trained models yet.</Text>
                    )}
                  </Flex>
                </Item>
              </TabPanels>
            </Tabs>
          </Content>
          <Footer>
            <Flex alignItems="center" gap="size-100">
              {isTrainingActive && (
                <ProgressCircle
                  aria-label="Training in progress"
                  isIndeterminate
                  size="S"
                />
              )}
              {message && (
                <Text>
                  {!isTrainingActive && messageIcon} {message}
                </Text>
              )}
              {isTrainingActive && !message && <Text>Training…</Text>}
            </Flex>
          </Footer>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Close
            </Button>
            <Button
              variant="accent"
              onPress={pressSubmit}
              isDisabled={
                !modelNames ||
                !selectedModelName ||
                activeTab !== "train" ||
                isTrainingActive
              }
            >
              Train
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}
