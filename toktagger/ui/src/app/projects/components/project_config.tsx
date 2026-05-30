import { Project, TaskType } from "@/types";
import {
  Button,
  ButtonGroup,
  Content,
  Dialog,
  DialogTrigger,
  Divider,
  Form,
  Heading,
  TextField,
  ComboBox,
  Item,
  RadioGroup,
  Radio,
  NumberField,
  Flex,
  Text,
  ToastQueue,
  ContextualHelp,
  Disclosure,
  DisclosureTitle,
  DisclosurePanel,
} from "@adobe/react-spectrum";
import AddCircle from "@spectrum-icons/workflow/AddCircle";
import Edit from "@spectrum-icons/workflow/EditCircle";
import { useState, useEffect } from "react";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { useAPISchema } from "@/app/contexts/apiSchema";
import { SchemaParser } from "@/schemaParser";

// Query strategies
const QueryStrategies = [
  { key: "sequential", value: "Sequential" },
  { key: "random", value: "Random" },
  { key: "uncertainty", value: "Uncertainty Sampling" },
];

// Tasks
const Tasks = Object.values(TaskType).map((task) => ({
  key: task,
  value: task,
}));

// Labels Form Component
const LabelsForm = ({
  label,
  defaultLabels,
  setLabels,
}: {
  label: string;
  defaultLabels: string[];
  setLabels: (labels: string[]) => void;
}) => {
  const [input, setInput] = useState<string>(defaultLabels.join(", "));

  useEffect(() => {
    setLabels(input.split(",").map((s) => s.trim()));
  }, [input, setLabels]);

  return (
    <Flex direction="row" gap="size-200" alignItems="center">
      <TextField width="100%" label={label} value={input} onChange={setInput} />
      <ContextualHelp placement="end bottom">
        <Content>
          <Text>
            {label} in a comma-separated format, e.g. &quot;class 1, class
            2&quot;. These labels will be used for {label.toLowerCase()}{" "}
            annotation.
          </Text>
        </Content>
      </ContextualHelp>
    </Flex>
  );
};

export function ProjectConfigEditor({
  project: _project,
  onModify,
}: {
  project?: Project;
  onModify?: () => void;
}) {
  const isEditing = !!_project;
  const titleText = isEditing ? "Edit Project" : "Create Project";
  const createText = isEditing ? "Save Changes" : "Create";
  const buttonText = isEditing ? "" : "Create";
  const icon = isEditing ? <Edit /> : <AddCircle />;

  const { schema } = useAPISchema();

  // Form state
  const [projectName, setProjectName] = useState<string>(_project?.name || "");
  const [task, setTask] = useState<string>(_project?.task || Tasks[0].key);
  const [queryStrategy, setQueryStrategy] = useState<string>(
    _project?.query_strategy || QueryStrategies[0].key,
  );
  const [dataLoader, setDataLoader] = useState<string | null>(
    _project?.data_loader || null,
  );
  const [dataLoaders, setDataLoaders] = useState<
    { key: string; value: string }[]
  >([]);

  // Optional time range fields
  const [timeMin, setTimeMin] = useState<number | null>(
    _project?.time_min || null,
  );
  const [timeMax, setTimeMax] = useState<number | null>(
    _project?.time_max || null,
  );
  const [minTimeStep, setMinTimeStep] = useState<number>(
    _project?.min_time_step || 0.0001,
  );

  // Label fields
  const [shotLabels, setShotLabels] = useState<string[]>(
    _project?.shot_labels || [],
  );
  const [timeRegionLabels, setTimeRegionLabels] = useState<string[]>(
    _project?.time_region_labels || [],
  );
  const [timePointLabels, setTimePointLabels] = useState<string[]>(
    _project?.time_point_labels || [],
  );
  const [boundingBoxLabels, setBoundingBoxLabels] = useState<string[]>(
    _project?.bounding_box_labels || [],
  );
  const [polygonLabels, setPolygonLabels] = useState<string[]>(
    _project?.polygon_labels || [],
  );
  const [videoBoundingBoxLabels, setVideoBoundingBoxLabels] = useState<
    string[]
  >(_project?.video_bounding_box_labels || []);

  useEffect(() => {
    const parser = new SchemaParser(schema);
    const defaultShotLabels = parser.parseDefaultShotLabels();
    if (shotLabels.length === 0) {
      setShotLabels(defaultShotLabels);
    }
    const defaultTimeRegionLabels = parser.parseDefaultTimeRegionLabels();
    if (timeRegionLabels.length === 0) {
      setTimeRegionLabels(defaultTimeRegionLabels);
    }
    const defaultTimePointLabels = parser.parseDefaultTimePointLabels();
    if (timePointLabels.length === 0) {
      setTimePointLabels(defaultTimePointLabels);
    }
    const defaultBoundingBoxLabels = parser.parseDefaultBoundingBoxLabels();
    if (boundingBoxLabels.length === 0) {
      setBoundingBoxLabels(defaultBoundingBoxLabels);
    }
    const defaultPolygonLabels = parser.parseDefaultPolygonLabels();
    if (polygonLabels.length === 0) {
      setPolygonLabels(defaultPolygonLabels);
    }
    const defaultVideoBoundingBoxLabels =
      parser.parseDefaultVideoBoundingBoxLabels();
    if (videoBoundingBoxLabels.length === 0) {
      setVideoBoundingBoxLabels(defaultVideoBoundingBoxLabels);
    }
  }, [
    schema,
    shotLabels,
    timeRegionLabels,
    timePointLabels,
    boundingBoxLabels,
    polygonLabels,
    videoBoundingBoxLabels,
  ]);

  // Fetch available data loaders on component mount
  useEffect(() => {
    async function fetchDataLoaders() {
      try {
        const response = await apiFetch(`${BACKEND_API_URL}/meta/dataloader`);
        if (response.ok) {
          const dataLoadersList = await response.json();
          const loaders = dataLoadersList.map((item: string) => ({
            key: item,
            value: item,
          }));
          setDataLoaders(loaders);
          if (loaders.length > 0 && !dataLoader) {
            setDataLoader(loaders[0].key);
          }
        } else {
          ToastQueue.negative(
            "Error fetching available Data Loaders from server.",
            { timeout: 3000 },
          );
        }
      } catch (error) {
        ToastQueue.negative(`Error fetching data loaders: ${error}`, {
          timeout: 3000,
        });
      }
    }
    fetchDataLoaders();
  }, [dataLoader]);

  const onFormSubmit = async (close: () => void) => {
    try {
      // Validate required fields
      if (!projectName.trim()) {
        ToastQueue.negative("Project name is required", { timeout: 3000 });
        return;
      }
      if (!dataLoader) {
        ToastQueue.negative("Data loader is required", { timeout: 3000 });
        return;
      }

      // Build project object
      const newProject: Partial<Project> = {
        name: projectName,
        task: task as TaskType,
        query_strategy: queryStrategy,
        data_loader: dataLoader,
        timestamp: new Date().toISOString(),
        time_min: timeMin,
        time_max: timeMax,
        min_time_step: minTimeStep,
        shot_labels: shotLabels,
        time_region_labels: timeRegionLabels,
        time_point_labels: timePointLabels,
        bounding_box_labels: boundingBoxLabels,
        polygon_labels: polygonLabels,
        video_bounding_box_labels: videoBoundingBoxLabels,
      };

      if (isEditing && _project?._id) {
        newProject._id = _project._id;
      }

      let url = `${BACKEND_API_URL}/projects`;
      let method: "POST" | "PUT" = "POST";
      if (isEditing && _project?._id) {
        url += `/${_project._id}`;
        method = "PUT";
      }

      // Create project via API
      const response = await apiFetch(url, {
        method: method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newProject),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Error creating project");
      }

      ToastQueue.positive("Project created successfully!", {
        timeout: 3000,
      });

      if (onModify) {
        onModify();
      }

      close();
    } catch (error) {
      ToastQueue.negative(`${error}`, { timeout: 3000 });
    }
  };

  return (
    <DialogTrigger>
      <Button
        aria-label={isEditing ? "Edit" : buttonText}
        variant={isEditing ? "accent" : "primary"}
      >
        {icon}
        {!isEditing ? <Text>{buttonText}</Text> : <></>}
      </Button>
      {(close) => (
        <Dialog>
          <Heading>{titleText}</Heading>
          <Divider />
          <Content>
            <Form maxWidth="size-6000">
              <TextField
                label="Project Name"
                isRequired
                value={projectName}
                onChange={setProjectName}
              />

              <ComboBox
                label="Task"
                items={Tasks}
                isRequired
                isDisabled={isEditing} // Disable task change when editing
                selectedKey={task}
                onSelectionChange={(key) =>
                  setTask(key ? String(key) : Tasks[0].key)
                }
              >
                {(item: Record<string, string>) => (
                  <Item key={item.key}>{item.value}</Item>
                )}
              </ComboBox>

              <ComboBox
                label="Data Loader"
                items={dataLoaders}
                isRequired
                selectedKey={dataLoader}
                onSelectionChange={(key) => {
                  setDataLoader(key ? String(key) : dataLoaders[0].key);
                }}
                isDisabled={isEditing} // Disable data loader change when editing
              >
                {(item: Record<string, string>) => (
                  <Item key={item.key}>{item.value}</Item>
                )}
              </ComboBox>

              <RadioGroup
                label="Query Strategy"
                isRequired
                value={queryStrategy}
                onChange={setQueryStrategy}
              >
                {QueryStrategies.map((item: Record<string, string>) => (
                  <Radio key={item.key} value={item.key}>
                    {item.value}
                  </Radio>
                ))}
              </RadioGroup>
              <Disclosure>
                {task !== TaskType.Video && (
                  <>
                    <DisclosureTitle>
                      <span style={{ fontSize: "0.9rem" }}>
                        Time Range Settings
                      </span>
                    </DisclosureTitle>
                    <DisclosurePanel>
                      <Flex direction="row" gap="size-200">
                        <NumberField
                          label="Time Min (s)"
                          value={timeMin ?? undefined}
                          onChange={(value) =>
                            setTimeMin(isNaN(value) ? null : value)
                          }
                          formatOptions={{
                            maximumFractionDigits: 10,
                          }}
                        />
                        <NumberField
                          label="Time Max (s)"
                          value={timeMax ?? undefined}
                          onChange={(value) =>
                            setTimeMax(isNaN(value) ? null : value)
                          }
                          formatOptions={{
                            maximumFractionDigits: 10,
                          }}
                        />
                      </Flex>

                      <NumberField
                        label="Min Time Step (s)"
                        value={minTimeStep}
                        onChange={setMinTimeStep}
                        formatOptions={{
                          maximumFractionDigits: 10,
                        }}
                      />
                    </DisclosurePanel>
                  </>
                )}
              </Disclosure>
              <Disclosure>
                <DisclosureTitle>
                  <span style={{ fontSize: "0.9rem" }}>
                    Annotation Label Settings
                  </span>
                </DisclosureTitle>
                <DisclosurePanel>
                  <LabelsForm
                    label="Shot Labels"
                    defaultLabels={shotLabels}
                    setLabels={setShotLabels}
                  />
                  {task !== TaskType.Video && (
                    <>
                      <LabelsForm
                        label="Time Region Labels"
                        defaultLabels={timeRegionLabels}
                        setLabels={setTimeRegionLabels}
                      />
                      <LabelsForm
                        label="Time Point Labels"
                        defaultLabels={timePointLabels}
                        setLabels={setTimePointLabels}
                      />
                    </>
                  )}
                  {/* {task === TaskType.Spectrogram ? (
                    <>
                      <LabelsForm
                        label="Bounding Box Labels"
                        defaultLabels={boundingBoxLabels}
                        setLabels={setBoundingBoxLabels}
                      />
                      <LabelsForm
                        label="Polygon Labels"
                        defaultLabels={polygonLabels}
                        setLabels={setPolygonLabels}
                      />
                    </>
                  ) : null} */}
                  {task === TaskType.Video && (
                    <LabelsForm
                      label="Video Bounding Box Labels"
                      defaultLabels={videoBoundingBoxLabels}
                      setLabels={setVideoBoundingBoxLabels}
                    />
                  )}
                </DisclosurePanel>
              </Disclosure>
            </Form>
          </Content>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Cancel
            </Button>
            <Button variant="primary" onPress={async () => onFormSubmit(close)}>
              {createText}
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}
