import { useState, useEffect, useRef } from "react";
import { z } from "zod/v4";
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
  Tabs,
  TabList,
  TabPanels,
  Key,
  TextField,
  ProgressCircle,
  TooltipTrigger,
  Tooltip,
  NumberField,
} from "@adobe/react-spectrum";
import FileWorkflow from "@spectrum-icons/workflow/FileWorkflow";
import CheckmarkCircle from "@spectrum-icons/workflow/CheckmarkCircle";
import DataAdd from "@spectrum-icons/workflow/DataAdd";
import Alert from "@spectrum-icons/workflow/Alert";
import { GitlabIcon, HuggingFaceIcon } from "@/app/utils";
import {
  Project,
  GitlabLoadForm,
  GitlabLoadFormSchema,
  LocalLoadForm,
  LocalLoadFormSchema,
  HuggingfaceLoadForm,
  HuggingfaceLoadFormSchema,
} from "@/types";
import {
  startLoadModelWeights,
  getLoadModelStatus,
  getModelTypes,
  getModelLoadTypes,
  getModelLoadAllowedIds,
} from "@/app/core";

type LoadMethod = "local" | "gitlab" | "hugging_face";

type LocalLoadProps = {
  form: LocalLoadForm;
  setForm: (formData: LocalLoadForm) => void;
  validationErrors: Record<string, string>;
};

type GitlabLoadProps = {
  form: GitlabLoadForm;
  setForm: (formData: GitlabLoadForm) => void;
  restrictedProjectId: boolean;
  validationErrors: Record<string, string>;
};

type HuggingfaceLoadProps = {
  form: HuggingfaceLoadForm;
  setForm: (formData: HuggingfaceLoadForm) => void;
  restrictedUserspace: boolean;
  validationErrors: Record<string, string>;
};

function LocalLoadTab({ form, setForm, validationErrors }: LocalLoadProps) {
  return (
    <Flex direction="column">
      <Text marginTop={"size-100"}>
        <em>
          Specify the path to the weights file to load, ensuring that the file
          has the correct permissions which allow it to be copied.
        </em>
      </Text>
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Model Weights Path"
        value={form.weights_path}
        validationState={
          "weights_path" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.weights_path ?? ""}
        onChange={(weights_path) =>
          setForm({
            ...form,
            weights_path,
          })
        }
      />
    </Flex>
  );
}

function GitlabLoadTab({
  form,
  setForm,
  restrictedProjectId,
  validationErrors,
}: GitlabLoadProps) {
  return (
    <Flex direction="column">
      <Text marginTop={"size-100"}>
        <em>Load model weights from the Gitlab ML Model Registry.</em>
      </Text>
      <NumberField
        marginTop={"size-100"}
        width={"100%"}
        label="Project ID"
        value={form.gitlab_project_id ?? undefined}
        validationState={
          "gitlab_project_id" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.gitlab_project_id ?? ""}
        onChange={(gitlab_project_id) =>
          setForm({
            ...form,
            gitlab_project_id,
          })
        }
        description={
          restrictedProjectId
            ? "Gitlab Project ID is configured on the server."
            : "The ID of the Gitlab project whose ML Model Registry will be connected to."
        }
        isDisabled={restrictedProjectId}
      />
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Model Name"
        value={form.model_name}
        validationState={
          "model_name" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.model_name ?? ""}
        onChange={(model_name) =>
          setForm({
            ...form,
            model_name,
          })
        }
        description="The name of the ML Model stored in the registry to download weights for."
      />
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Model Version"
        value={form.model_version}
        validationState={
          "model_version" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.model_version ?? ""}
        onChange={(model_version) =>
          setForm({
            ...form,
            model_version,
          })
        }
        description="Optional: The semantic version of the model to download, eg v1.0.0"
      />
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Weights Path"
        value={form.weights_path}
        validationState={
          "weights_path" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.weights_path ?? ""}
        onChange={(weights_path) =>
          setForm({
            ...form,
            weights_path,
          })
        }
        description="The path to the weights artifact within the model registry."
      />
    </Flex>
  );
}

function HuggingfaceLoadTab({
  form,
  setForm,
  restrictedUserspace,
  validationErrors,
}: HuggingfaceLoadProps) {
  return (
    <Flex direction="column">
      <Text marginTop={"size-100"}>
        <em>Load model weights from HuggingFace.</em>
      </Text>
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Userspace or Organisation"
        value={form.huggingface_userspace ?? undefined}
        validationState={
          "huggingface_userspace" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.huggingface_userspace ?? ""}
        onChange={(huggingface_userspace) =>
          setForm({
            ...form,
            huggingface_userspace,
          })
        }
        description={
          restrictedUserspace
            ? "HuggingFace userspace or organisation is configured on the server."
            : "The ID of the HuggingFace userspace or organisation which will be connected to."
        }
        isDisabled={restrictedUserspace}
      />
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Model Name"
        value={form.model_name}
        validationState={
          "model_name" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.model_name ?? ""}
        onChange={(model_name) =>
          setForm({
            ...form,
            model_name,
          })
        }
        description="The name of the project stored in HuggingFace to download weights for."
      />
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Model Version"
        value={form.model_version}
        validationState={
          "model_version" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.model_version ?? ""}
        onChange={(model_version) =>
          setForm({
            ...form,
            model_version,
          })
        }
        description="Optional: The semantic version or revision of the model to download, eg v1.0.0."
      />
      <TextField
        marginTop={"size-100"}
        width={"100%"}
        label="Weights Path"
        value={form.weights_path}
        validationState={
          "weights_path" in validationErrors ? "invalid" : undefined
        }
        errorMessage={validationErrors.weights_path ?? ""}
        onChange={(weights_path) =>
          setForm({
            ...form,
            weights_path,
          })
        }
        description="The path to the weights artifact within the HuggingFace project."
      />
    </Flex>
  );
}

export function ModelLoadModal({
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
  const pollingModelName = useRef<string | null>(null);
  const [loadMethods, setLoadMethods] = useState<string[] | null>(null);
  const [restrictedRemoteId, setRestrictedRemoteId] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [taskId, setTaskId] = useState<string | null>(null);

  const [selectedTab, setSelectedTab] = useState<LoadMethod | null>(null);

  const [localForm, setLocalForm] = useState<LocalLoadForm>({
    weights_path: "",
  });

  const [gitlabForm, setGitlabForm] = useState<GitlabLoadForm>({
    gitlab_project_id: 0,
    model_name: "",
    weights_path: "",
    model_version: undefined,
  });

  const [huggingfaceForm, setHuggingfaceForm] = useState<HuggingfaceLoadForm>({
    model_name: "",
    weights_path: "",
    model_version: undefined,
    huggingface_userspace: "",
  });

  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});

  const submitLoadJob = async () => {
    if (!selectedModelName || !selectedTab || !project._id) {
      return;
    }
    let params: LocalLoadForm | GitlabLoadForm;
    let valid:
      | z.ZodSafeParseResult<LocalLoadForm>
      | z.ZodSafeParseResult<GitlabLoadForm>;

    if (selectedTab == "local") {
      params = localForm;
      valid = LocalLoadFormSchema.safeParse(params);
    } else if (selectedTab == "gitlab") {
      params = gitlabForm;
      valid = GitlabLoadFormSchema.safeParse(params);
    } else if (selectedTab == "hugging_face") {
      params = huggingfaceForm;
      valid = HuggingfaceLoadFormSchema.safeParse(params);
    } else {
      throw new Error("Unrecognised model load type!");
    }

    if (!valid.success) {
      setMessage("Invalid parameters provided!");
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);

      const errors = Object.fromEntries(
        valid.error.issues.map((issue) => [
          String(issue.path[0]),
          issue.message,
        ]),
      );

      setValidationErrors(errors);
      return;
    }
    setValidationErrors({});
    let response: Response;
    response = await startLoadModelWeights(
      project._id,
      selectedTab,
      selectedModelName,
      params,
    );
    const payload = await response.json();

    if (response.ok) {
      setIsLoading(true);
      setTaskId(payload.task_id);
      pollingModelName.current = selectedModelName;
      setMessage(null);
    } else if (response.status == 422) {
      setMessage("Invalid parameters provided!");
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      console.log(payload.detail);
      const errors = Object.fromEntries(
        payload.detail.map((issue: Record<string, string | string[]>) => [
          issue.loc.at(-1),
          issue.msg,
        ]),
      );
      setValidationErrors(errors);
      return;
    } else {
      console.log(payload.detail);
      setMessage(payload.detail);
      setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
    }
  };

  useEffect(() => {
    if (!taskId || !project._id || !pollingModelName.current) return;

    let pollCounter = 0;
    // Poll for result from GET predictions endpoint
    const interval = setInterval(async () => {
      if (pollingModelName.current == null) {
        clearInterval(interval);
        setIsLoading(false);
        return;
      }
      const response = await getLoadModelStatus(
        project._id,
        pollingModelName.current,
        taskId,
      );
      const payload = await response.json();

      if (response.status === 202) {
        // Load check queued but not done yet, so continue to poll
        pollCounter += 1;
        if (pollCounter > 60) {
          setMessage(
            "Timed out while loading model - check models tab to see current status.",
          );
          setMessageIcon(
            <Alert aria-label="Timeout" color="notice" size="S" />,
          );
          clearInterval(interval);
          setIsLoading(false);
        }
      } else if (response.ok && payload === true) {
        setMessage("Model loaded successfully!");
        setMessageIcon(
          <CheckmarkCircle aria-label="Success" color="positive" size="S" />,
        );
        clearInterval(interval);
        setIsLoading(false);
      } else {
        setMessage(
          payload === false ? "Model failed to load!" : payload.detail,
        );
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
        clearInterval(interval);
        setIsLoading(false);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [project._id, taskId]);

  useEffect(() => {
    if (!selectedTab) {
      return;
    }
    (async () => {
      const response = await getModelLoadAllowedIds(selectedTab as string);
      if (response.ok) {
        const data = await response.json();
        if (!data) {
          setRestrictedRemoteId(false);
          return;
        }
        setRestrictedRemoteId(true);
        if (selectedTab === "gitlab") {
          setGitlabForm((prev) => ({
            ...prev,
            gitlab_project_id: Number(data),
          }));
        } else if (selectedTab === "hugging_face") {
          setHuggingfaceForm((prev) => ({
            ...prev,
            huggingface_userspace: data as string,
          }));
        }
      } else {
        const errorMessage = await response.json();
        setMessage(errorMessage.detail);
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      }
    })();
  }, [selectedTab]);

  useEffect(() => {
    if (!modalOpen) {
      return;
    }
    (async () => {
      const modelTypesResponse = await getModelTypes(project.task);
      const modelLoadResponse = await getModelLoadTypes();
      if (modelTypesResponse.ok) {
        const data = await modelTypesResponse.json();
        const modelTypes = data as string[];
        setModelNames(modelTypes);
      } else {
        const errorMessage = await modelTypesResponse.json();
        setMessage(errorMessage.detail);
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      }
      if (modelLoadResponse.ok) {
        const data = await modelLoadResponse.json();
        const loadMethods = data as LoadMethod[];
        setLoadMethods(loadMethods);
        setSelectedTab(loadMethods?.[0] ?? null);
      } else {
        const errorMessage = await modelLoadResponse.json();
        setMessage(errorMessage.detail);
        setMessageIcon(<Alert aria-label="Failed" color="negative" size="S" />);
      }
    })();
  }, [modalOpen, project.task]);

  return (
    <DialogTrigger onOpenChange={(isOpen) => setModalOpen(isOpen)}>
      <TooltipTrigger delay={350} placement="bottom">
        <ActionButton isDisabled={!isEnabled} aria-label="Load ML Model">
          <FileWorkflow />
        </ActionButton>
        <Tooltip>"Load Pretrained Weights"</Tooltip>
      </TooltipTrigger>
      {(close) => (
        <Dialog>
          <Heading>
            <Flex alignItems="center" gap="size-100">
              <FileWorkflow size="S" />
              <Text>Load Pretrained Model Weights</Text>
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
            {loadMethods && (
              <Tabs
                aria-label="ML Model Tabs"
                selectedKey={selectedTab}
                onSelectionChange={(key) =>
                  setSelectedTab(String(key) as LoadMethod)
                }
              >
                <TabList>
                  {loadMethods?.includes("local") ? (
                    <Item key="local" aria-label="Use Local File Tab">
                      <DataAdd />
                      <Text>Use Local File</Text>
                    </Item>
                  ) : null}
                  {loadMethods?.includes("gitlab") ? (
                    <Item key="gitlab" aria-label="From Gitlab Tab">
                      <GitlabIcon />
                      <Text>From Gitlab</Text>
                    </Item>
                  ) : null}
                  {loadMethods?.includes("hugging_face") ? (
                    <Item key="hugging_face" aria-label="From HuggingFace Tab">
                      <HuggingFaceIcon />
                      <Text>From HuggingFace</Text>
                    </Item>
                  ) : null}
                </TabList>
                <TabPanels>
                  {loadMethods?.includes("local") ? (
                    <Item key="local" aria-label="Load from local file form">
                      <LocalLoadTab
                        form={localForm}
                        setForm={setLocalForm}
                        validationErrors={validationErrors}
                      />
                    </Item>
                  ) : null}
                  {loadMethods?.includes("gitlab") ? (
                    <Item key="gitlab" aria-label="Load from Gitlab form">
                      <GitlabLoadTab
                        form={gitlabForm}
                        setForm={setGitlabForm}
                        restrictedProjectId={restrictedRemoteId}
                        validationErrors={validationErrors}
                      />
                    </Item>
                  ) : null}
                  {loadMethods?.includes("hugging_face") ? (
                    <Item
                      key="hugging_face"
                      aria-label="Load from HuggingFace form"
                    >
                      <HuggingfaceLoadTab
                        form={huggingfaceForm}
                        setForm={setHuggingfaceForm}
                        restrictedUserspace={restrictedRemoteId}
                        validationErrors={validationErrors}
                      />
                    </Item>
                  ) : null}
                </TabPanels>
              </Tabs>
            )}
          </Content>
          <Footer>
            {message && (
              <Text>
                {messageIcon} {message}
              </Text>
            )}
            {isLoading && (
              <ProgressCircle aria-label="Loading…" isIndeterminate />
            )}
          </Footer>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Close
            </Button>
            <Button
              variant="accent"
              onPress={submitLoadJob}
              isDisabled={!selectedModelName || isLoading}
            >
              Submit
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}
