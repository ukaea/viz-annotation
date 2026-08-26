"use client";
import { useEffect, useState, useCallback } from "react";
import {
  Cell,
  Column,
  Row,
  TableView,
  TableBody,
  TableHeader,
  Item,
  Flex,
  Button,
  ActionButton,
  Picker,
  SearchField,
  DialogTrigger,
  Dialog,
  Heading,
  Divider,
  Content,
  ButtonGroup,
  Checkbox,
  View,
  ContextualHelp,
  Footer,
  Link,
  Provider,
  defaultTheme,
  ToastContainer,
  Text,
} from "@adobe/react-spectrum";
import { SortDescriptor } from "@react-types/shared";
import { AddSamplesEditor } from "./components/add_samples";
import { ProjectMembersDialog } from "./components/members";
import {
  getSamples,
  getProject,
  deleteSample,
  deleteSamples,
  ApiError,
} from "@/app/core";
import ErrorView from "@/app/views/error";
import ForbiddenView from "@/app/views/forbidden";
import { ModelTrainModal } from "@/app/components/tools/modelTrain";
import { ModelPredictModal } from "@/app/components/tools/modelPredict";
import { ModelLoadModal } from "@/app/components/tools/modelLoad";
import Delete from "@spectrum-icons/workflow/Delete";
import type { Project, Sample } from "@/types";
import { useParams } from "react-router-dom";
import { ImportButton } from "@/app/components/tools/import";
import { ExportButton } from "@/app/components/tools/export";
import { JumpToNextButton } from "@/app/components/tools/nav";
import { useServerHealth } from "@/app/contexts/healthContext";
import { useProjectRole } from "@/app/hooks/useProjectRole";
import { useBreadcrumbs } from "@/app/contexts/BreadcrumbContext";

type SamplesTableProps = {
  project_id: string;
  samples: Sample[];
  sortDescriptor: SortDescriptor;
  onSortChange: (sort: SortDescriptor) => void;
  onModify?: () => void;
  canManageSamples: boolean;
};

const SamplesTable = ({
  project_id,
  samples,
  sortDescriptor,
  onSortChange,
  onModify,
  canManageSamples,
}: SamplesTableProps) => {
  const rows = samples.map(({ _id, ...rest }) => ({
    ...rest,
    id: _id,
  }));

  return (
    <Flex height="size-5000" width="100%" direction="column">
      <TableView
        flex
        aria-label="Samples"
        selectionMode="none"
        selectionStyle="highlight"
        sortDescriptor={sortDescriptor}
        onSortChange={onSortChange}
      >
        <TableHeader>
          <Column key="shot_id" allowsSorting>
            Shot ID
          </Column>
          <Column key="_id" allowsSorting>
            Date Created
          </Column>
          <Column key="validated_annotations" allowsSorting>
            Validated
          </Column>
          <Column key="actions">Actions</Column>
        </TableHeader>
        <TableBody items={rows}>
          {(item) => (
            <Row
              href={`/ui/projects/${project_id}/samples/${item["id"]}?sortColumn=${sortDescriptor.column}&sortDirection=${sortDescriptor.direction}`}
            >
              <Cell>{item["shot_id"]}</Cell>
              <Cell>{item["timestamp"]}</Cell>
              <Cell>
                <Checkbox
                  aria-label="Validated Annotations"
                  isSelected={item["validated_annotations"]}
                  isReadOnly={true}
                />
              </Cell>
              <Cell>
                <Flex direction="row" gap="size-100">
                  <DialogTrigger>
                    <Button
                      aria-label="Delete"
                      variant="negative"
                      isDisabled={!canManageSamples}
                    >
                      <Delete />
                    </Button>
                    {(close) => (
                      <Dialog>
                        <Heading>Confirm Deletion</Heading>
                        <Divider />
                        <Content>
                          Are you sure you want to delete sample with Shot ID{" "}
                          <strong>{item["shot_id"]}</strong>? You will also lose{" "}
                          <strong>all annotations</strong> associated with this
                          sample. This action cannot be undone.
                        </Content>
                        <ButtonGroup>
                          <Button variant="secondary" onPress={close}>
                            Cancel
                          </Button>
                          <Button
                            variant="negative"
                            onPress={async () => {
                              if (item["id"] == null) {
                                return;
                              }
                              await deleteSample(project_id, item["id"]);
                              onModify?.();
                              close();
                            }}
                          >
                            Delete
                          </Button>
                        </ButtonGroup>
                      </Dialog>
                    )}
                  </DialogTrigger>
                </Flex>
              </Cell>
            </Row>
          )}
        </TableBody>
      </TableView>
    </Flex>
  );
};

export default function ProjectView() {
  const { project_id } = useParams();
  const { isAdmin, canAnnotate } = useProjectRole(project_id);
  const hasId = project_id !== undefined;

  const [samplesPerPage, setSamplesPerPage] = useState<number>(10);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [shotId, setShotId] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [sortDescriptor, setSortDescriptor] = useState<SortDescriptor>({
    column: "shot_id",
    direction: "ascending",
  });
  const [samples, setSamples] = useState<Sample[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [loadError, setLoadError] = useState<{
    forbidden: boolean;
    message: string;
  } | null>(null);
  const { modelsEnabled } = useServerHealth();
  useBreadcrumbs(
    project
      ? [
          { key: "projects", label: "Projects", href: "/ui/projects" },
          {
            key: "project",
            label: `Project: ${project.name}`,
            href: `/ui/projects/${project._id}`,
          },
        ]
      : loadError
        ? [
            { key: "projects", label: "Projects", href: "/ui/projects" },
            {
              key: "denied",
              label: loadError.forbidden ? "Access Denied" : "Error",
            },
          ]
        : // Spectrum renders the last breadcrumb as the (unclickable) current
          // page, so a single "Projects" crumb here can't act as a link.
          [{ key: "projects", label: "Projects", href: "/ui/projects" }],
  );

  const refreshSamples = useCallback(async () => {
    if (!hasId) {
      return;
    }
    try {
      const samples = await getSamples(
        sortDescriptor,
        project_id,
        currentPage,
        samplesPerPage,
        shotId,
      );
      setSamples(samples);
      const project = await getProject(project_id);
      setProject(project);
      setLoadError(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        // A non-member is told plainly that access is refused, rather than that the
        // project does not exist. This does confirm the project exists to anyone who
        // guesses its ID, which is the accepted trade for an actionable message.
        setLoadError({
          forbidden: true,
          message: err.message || "You are not a member of this project.",
        });
      } else if (err instanceof ApiError && err.status === 404) {
        setLoadError({ forbidden: false, message: "Project not found." });
      } else {
        setLoadError({
          forbidden: false,
          message:
            err instanceof ApiError ? err.message : "Failed to load project.",
        });
      }
    }
  }, [project_id, shotId, currentPage, samplesPerPage, sortDescriptor, hasId]);

  useEffect(() => {
    refreshSamples();
  }, [
    refreshSamples,
    project_id,
    shotId,
    currentPage,
    samplesPerPage,
    sortDescriptor,
    hasId,
  ]);

  if (loadError) {
    return loadError.forbidden ? (
      <ForbiddenView message={loadError.message} />
    ) : (
      <ErrorView message={loadError.message} />
    );
  }

  if (!project || !hasId) {
    return;
  }

  if (!samples) {
    return;
  }

  const onSortChange = (newSortDescriptor: SortDescriptor) => {
    setSortDescriptor(newSortDescriptor);
  };

  const onSearchSubmit = (newValue: string) => {
    if (/^[0-9]*$/.test(newValue)) {
      setErrorMessage("");
      setShotId(newValue);
      setCurrentPage(1);
    } else {
      setErrorMessage("Please enter a number.");
    }
  };

  return (
    <div className="h-full">
      <div className="relative w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400 dark:from-gray-700 dark:via-gray-800 dark:to-gray-900">
        <div className="w-full md:w-4/5 p-6 bg-white/60 dark:bg-gray-800/60 text-gray-800 dark:text-gray-100 rounded-lg shadow-lg backdrop-blur-sm">
          <h1 className="text-2xl font-bold mb-4">Samples</h1>
          <Provider theme={defaultTheme}>
            <ToastContainer placement="top" />
            <View overflow="auto">
              <Flex
                direction="row"
                marginY="size-100"
                gap="size-100"
                alignItems="end"
                justifyContent="space-between"
                wrap
              >
                <Flex
                  gap="size-100"
                  alignItems="end"
                  justifyContent="start"
                  wrap
                >
                  <AddSamplesEditor
                    project={project}
                    onModify={refreshSamples}
                    canManageSamples={isAdmin}
                  />
                  {project_id && (
                    <ProjectMembersDialog
                      projectId={project_id}
                      isProjectAdmin={isAdmin}
                    />
                  )}
                  <ImportButton project={project} canAnnotate={canAnnotate} />
                  <ExportButton project={project} />
                  <JumpToNextButton
                    project={project}
                    sortDescriptor={sortDescriptor}
                  />
                  <DialogTrigger>
                    <ActionButton isQuiet isDisabled={!isAdmin}>
                      <Delete />
                      <Text>Clear Samples</Text>
                    </ActionButton>
                    {(close) => (
                      <Dialog>
                        <Heading>Confirm Clear All Samples</Heading>
                        <Divider />
                        <Content>
                          Are you sure you want to delete{" "}
                          <strong>all samples</strong> in this project? You will
                          lose <strong>all annotations</strong> associated with
                          the samples as well. This action cannot be undone.
                        </Content>
                        <ButtonGroup>
                          <Button variant="secondary" onPress={close}>
                            Cancel
                          </Button>
                          <Button
                            variant="negative"
                            onPress={async () => {
                              if (!project_id) {
                                return;
                              }
                              await deleteSamples(project_id);
                              refreshSamples();
                              close();
                            }}
                          >
                            Clear All
                          </Button>
                        </ButtonGroup>
                      </Dialog>
                    )}
                  </DialogTrigger>
                </Flex>
                <Flex gap="size-100" alignItems="end" wrap>
                  <Flex direction="row" gap={"size-100"}>
                    {/* Training/loading/predicting all write to the project, so
                    a viewer gets the same disabled state the backend already
                    enforces (require_project_annotator on these endpoints). */}
                    <ModelTrainModal
                      project={project}
                      isEnabled={modelsEnabled && canAnnotate}
                    ></ModelTrainModal>
                    <ModelLoadModal
                      project={project}
                      isEnabled={modelsEnabled && canAnnotate}
                    ></ModelLoadModal>
                    <ModelPredictModal
                      project={project}
                      isEnabled={modelsEnabled && canAnnotate}
                    ></ModelPredictModal>
                    <ContextualHelp
                      placement="top end"
                      aria-label="ML Model Help"
                    >
                      <Heading>
                        {modelsEnabled
                          ? "ML Model Controls"
                          : "ML Models Disabled"}
                      </Heading>
                      <Content>
                        {modelsEnabled
                          ? "Use these inputs to train / load and make predictions with Machine Learning models. You can define custom ML models for your datasets using the TokTagger Python module."
                          : "Model tools are disabled due to missing dependencies on the server."}
                      </Content>
                      <Footer>
                        <Link href="https://ukaea.github.io/toktagger/custom_models/">
                          Learn more about ML models in TokTagger
                        </Link>
                      </Footer>
                    </ContextualHelp>
                  </Flex>
                  <SearchField
                    label="Search By Shot ID"
                    width="size-1700"
                    onSubmit={onSearchSubmit}
                    validationState={errorMessage ? "invalid" : undefined}
                    errorMessage={errorMessage}
                  />
                </Flex>
              </Flex>
            </View>
            <SamplesTable
              project_id={project_id}
              samples={samples}
              sortDescriptor={sortDescriptor}
              onSortChange={onSortChange}
              onModify={refreshSamples}
              canManageSamples={isAdmin}
            />
            <div className="flex items-center justify-between pl-4 pr-4">
              <Button
                variant="primary"
                onPress={() => setCurrentPage((p) => p - 1)}
                isDisabled={currentPage === 1}
              >
                Previous
              </Button>
              <div className="flex items-center justify-center gap-8 pb-2">
                <p> Page: {currentPage} </p>
                <Picker
                  label="Samples per Page:"
                  onSelectionChange={(selectedKey) => {
                    if (selectedKey != null) {
                      setSamplesPerPage(Number(selectedKey) || 10);
                      setCurrentPage(1);
                    }
                  }}
                  defaultSelectedKey="10"
                >
                  <Item key="5">5</Item>
                  <Item key="10">10</Item>
                  <Item key="25">25</Item>
                  <Item key="50">50</Item>
                </Picker>
              </div>
              <Button
                variant="primary"
                onPress={() => setCurrentPage((p) => p + 1)}
                isDisabled={samples.length < samplesPerPage}
              >
                Next
              </Button>
            </div>
          </Provider>
        </div>
      </div>
    </div>
  );
}
