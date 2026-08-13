"use client";

import {
  Button,
  Content,
  ContextualHelp,
  Divider,
  Flex,
  Heading,
  Item,
  Picker,
  Text,
  ToggleButton,
  Tooltip,
  TooltipTrigger,
  View,
} from "@adobe/react-spectrum";
import {
  useTimeSeriesState,
  useTimeSeriesActions,
} from "@/app/contexts/TimeSeriesContext";
import { useEffect, useState } from "react";
import { TimeSeriesAnnotationType } from "@/types";
import { useSample } from "@/app/contexts/SampleContext";

const categoryAllocsKey = (projectId: string) =>
  `ts-category-allocs-${projectId}`;

// Reads the saved category -> label allocations, discarding corrupt storage.
// Individual labels are still checked against the project's categories below.
function readSavedAllocations(projectId: string): Record<string, string> {
  const savedRaw = sessionStorage.getItem(categoryAllocsKey(projectId));
  if (!savedRaw) return {};

  try {
    const parsed: unknown = JSON.parse(savedRaw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      !Array.isArray(parsed)
    ) {
      return parsed as Record<string, string>;
    }
  } catch {
    // Malformed JSON - fall through and discard.
  }

  sessionStorage.removeItem(categoryAllocsKey(projectId));
  return {};
}

export const AnnotationToolbar = () => {
  const {
    editMode,
    toolingCallbacks,
    categories,
    activeAnnotationTool,
    canAnnotate,
  } = useTimeSeriesState();
  const { setEditMode, setAnnotationTool } = useTimeSeriesActions();
  const { project } = useSample();
  const projectId = project?._id;

  const [categoryAllocations, setCategoryAllocations] = useState<
    Map<TimeSeriesAnnotationType, string>
  >(new Map());

  const [firstTimeEdit, setFirstTimeEdit] = useState(
    () => localStorage.getItem("ts-help-seen") !== "true",
  );
  const [contextHelpManualOpen, setContextHelpManualOpen] = useState<
    boolean | undefined
  >(undefined);

  const modeVariant: "accent" | "primary" = editMode ? "accent" : "primary";
  const modeText = editMode ? "Edit Mode" : "View Mode";

  useEffect(() => {
    if (!projectId) return;
    const saved = readSavedAllocations(projectId);

    const categoryMap: Map<TimeSeriesAnnotationType, string> = new Map();
    categories.forEach((category) => {
      if (!categoryMap.has(category.type)) {
        const savedLabel = saved[category.type];
        const savedValid =
          savedLabel !== undefined &&
          [...categories.values()].some(
            (c) => c.type === category.type && c.label === savedLabel,
          );
        categoryMap.set(
          category.type,
          savedValid ? savedLabel : category.label,
        );
      }
    });
    setCategoryAllocations(categoryMap);
  }, [categories, projectId]);

  useEffect(() => {
    if (!projectId || categoryAllocations.size === 0) return;
    const record: Record<string, string> = {};
    categoryAllocations.forEach((label, type) => {
      record[type] = label;
    });
    sessionStorage.setItem(
      categoryAllocsKey(projectId),
      JSON.stringify(record),
    );
  }, [categoryAllocations, projectId]);

  return (
    <View
      width="size-3000"
      flexShrink={0}
      marginTop="size-200"
      data-testid="annotation-toolbar"
    >
      <Flex direction="column" alignItems="center" gap="size-150">
        <h1 className="text-2xl font-bold">Annotation Toolbar</h1>
        <TooltipTrigger placement="left">
          <Button
            width="size-1600"
            variant={modeVariant}
            isDisabled={!canAnnotate}
            onPress={() => {
              setEditMode(!editMode);
              if (firstTimeEdit) {
                setContextHelpManualOpen(true);
              }
            }}
          >
            {modeText}
          </Button>
          <Tooltip>
            {canAnnotate
              ? `Click to enter ${
                  editMode
                    ? "view mode - annotations disabled"
                    : "edit mode - annotations enabled"
                } (shortcut: e)`
              : "You have view-only access to this project — annotations cannot be edited."}
          </Tooltip>
        </TooltipTrigger>
        <Divider size="S" marginX="size-200" />
        <h1 className="text-xl font-bold">Tools</h1>
        <Flex direction="column" alignItems="center" gap="size-100">
          {[...toolingCallbacks.keys()].map((info) => {
            const toolActive = info === activeAnnotationTool?.type;
            return (
              <Flex
                key={info}
                direction="column"
                alignItems="center"
                gap="size-100"
              >
                <TooltipTrigger placement="left">
                  <ToggleButton
                    width="size-1600"
                    isDisabled={!editMode}
                    isSelected={toolActive}
                    onPress={() => {
                      if (toolActive) {
                        setAnnotationTool(null);
                        return;
                      }
                      setAnnotationTool({
                        type: info,
                        label: categoryAllocations.get(info)!,
                      });
                    }}
                  >
                    {info}
                  </ToggleButton>
                  <Tooltip>{`Click to ${toolActive ? "deactivate" : "activate"} ${info} tooling`}</Tooltip>
                </TooltipTrigger>
                {toolActive && (
                  <Picker
                    data-testid="select-annotation-label"
                    label="Select label"
                    width="size-2400"
                    isDisabled={!editMode}
                    items={categories
                      .values()
                      .filter((category) => category.type === info)}
                    selectedKey={categoryAllocations.get(info)}
                    onSelectionChange={(key) => {
                      setCategoryAllocations((prev) => {
                        const newMap = new Map(prev);
                        newMap.set(info, key as string);
                        return newMap;
                      });
                      setAnnotationTool({ type: info, label: key as string });
                    }}
                  >
                    {(item) => <Item key={item.label}>{item.label}</Item>}
                  </Picker>
                )}
              </Flex>
            );
          })}
        </Flex>
      </Flex>
      <Flex direction="row" justifyContent="end" marginEnd="size-100">
        <ContextualHelp
          isOpen={firstTimeEdit ? contextHelpManualOpen : undefined}
          onOpenChange={() => {
            setFirstTimeEdit(false);
            localStorage.setItem("ts-help-seen", "true");
          }}
          aria-label="annotation-context-help"
        >
          <Heading>Annotation Toolbar</Heading>
          <Content>
            <Text>
              Use the top button to switch between <b>edit mode</b> and{" "}
              <b>view mode</b>. Press <b>e</b> to toggle quickly.
              <br />
              <br />
              Activate the desired tool using the list of buttons - use the
              dropdown menu that appears to select the relevant label.
              <br />
              <br />
              When a tool is active, new annotations can be added using{" "}
              <b>ctrl+drag</b>.
            </Text>
          </Content>
        </ContextualHelp>
      </Flex>
    </View>
  );
};
