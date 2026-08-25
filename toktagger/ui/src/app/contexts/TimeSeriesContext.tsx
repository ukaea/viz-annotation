"use client";

import {
  Annotation,
  SelectionRange,
  TimeSeriesAnnotation,
  TimeSeriesAnnotationType,
  TimeSeriesCategory,
  TimeSeriesToolDefinition,
  ToolingCallbacks,
} from "@/types";
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { v4 as uuidv4 } from "uuid";
import { useSample } from "./SampleContext";
import { useAuth } from "./AuthContext";
import { useProjectRole } from "@/app/hooks/useProjectRole";
import {
  convertRawAnnotationsToTimeSeries,
  convertTimeSeriesToRawAnnotations,
  isTimeSeriesAnnotation,
  randomColor,
} from "../utils";
import { Item, ItemParams, Menu, Submenu } from "react-contexify";

type TimeSeriesActions = {
  setAnnotations: (annotations: TimeSeriesAnnotation[]) => void;
  createAnnotation: (
    type: TimeSeriesAnnotationType,
    label: string,
  ) => TimeSeriesAnnotation;
  addAnnotation: (annotation: TimeSeriesAnnotation) => void;
  removeAnnotation: (id: string) => void;
  updateAnnotation: (annotation: TimeSeriesAnnotation) => void;
  getAnnotation: (id: string) => TimeSeriesAnnotation | null;
  setAnnotationTool: (tool: TimeSeriesToolDefinition | null) => void;
  registerTooling: (
    type: TimeSeriesAnnotationType,
    callbacks: ToolingCallbacks,
  ) => void;
  triggerUpdate: () => void;
  selectAnnotations: (ids: string[]) => void;
  findSelectedAnnotations: (range: SelectionRange | null) => void;
  setEditMode: (turnOn: boolean) => void;
  setOngoingAction: (state: boolean) => void;
};

type TimeSeriesState = {
  annotations: TimeSeriesAnnotation[];
  activeAnnotationTool: TimeSeriesToolDefinition | null;
  toolingCallbacks: Map<TimeSeriesAnnotationType, ToolingCallbacks>;
  forceUpdate: number;
  isDrawing: boolean;
  ongoingAction: boolean;
  categories: Map<string, TimeSeriesCategory>;
  editMode: boolean;
  canAnnotate: boolean;
};

const TimeSeriesActionsContext = createContext<TimeSeriesActions | null>(null);
const TimeSeriesStateContext = createContext<TimeSeriesState | null>(null);

export const useTimeSeriesActions = () => {
  const context = useContext(TimeSeriesActionsContext);
  if (!context) {
    throw new Error(
      "useTimeSeriesActions must be used within a TimeSeriesProvider",
    );
  }
  return context;
};

export const useTimeSeriesState = () => {
  const context = useContext(TimeSeriesStateContext);
  if (!context) {
    throw new Error(
      "useTimeSeriesState must be used within a TimeSeriesProvider",
    );
  }
  return context;
};

export const TIME_SERIES_ANNOTATION_MENU = "time-series-annotation-menu";

function isEditableEventTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLSelectElement) return true;
  if (target instanceof HTMLInputElement) {
    return target.type !== "checkbox" && target.type !== "radio";
  }
  return false;
}

const activeToolKey = (projectId: string) => `ts-active-tool-${projectId}`;

// Reads a persisted tool, discarding anything that isn't a well-formed
// TimeSeriesToolDefinition so that corrupt storage cannot throw during render.
// The label is not checked here - that needs the project's categories, which
// are not loaded yet at this point.
function readSavedTool(projectId: string): TimeSeriesToolDefinition | null {
  if (!projectId) return null;
  const saved = sessionStorage.getItem(activeToolKey(projectId));
  if (!saved) return null;

  try {
    const parsed: unknown = JSON.parse(saved);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      typeof (parsed as TimeSeriesToolDefinition).label === "string" &&
      Object.values(TimeSeriesAnnotationType).includes(
        (parsed as TimeSeriesToolDefinition).type,
      )
    ) {
      return parsed as TimeSeriesToolDefinition;
    }
  } catch {
    // Malformed JSON - fall through and discard.
  }

  sessionStorage.removeItem(activeToolKey(projectId));
  return null;
}

export const TimeSeriesProvider = ({
  signalName = null,
  children,
}: {
  // Binds annotations created here to a signal; null for single-signal views.
  signalName?: string | null;
  children: React.ReactNode;
}) => {
  const {
    annotations: rawAnnotations,
    setAnnotations: setRawAnnotations,
    project,
  } = useSample();

  // project is guaranteed non-null here: TimeSeriesProvider is only rendered
  // after SampleView confirms project is loaded.
  const projectId = project?._id ?? "";
  const { canAnnotate } = useProjectRole(project?._id);
  const { user } = useAuth();

  const [annotations, setAnnotations] = useState<TimeSeriesAnnotation[]>([]);
  const [toolingCallbacks, setToolingCallbacks] = useState<
    Map<TimeSeriesAnnotationType, ToolingCallbacks>
  >(new Map());
  const [activeTool, setActiveTool] = useState<TimeSeriesToolDefinition | null>(
    null,
  );
  // A restored tool cannot be applied on mount: tooling callbacks register from
  // child components and categories come from the project, so neither is
  // available yet. Hold it here until both are, then validate and apply.
  const [pendingTool, setPendingTool] =
    useState<TimeSeriesToolDefinition | null>(() => readSavedTool(projectId));
  const [updateCounter, setUpdateCounter] = useState(0);
  const [syncCounter, setSyncCounter] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [categories, setCategories] = useState<Map<string, TimeSeriesCategory>>(
    new Map(),
  );
  const [editMode, setEditModeRaw] = useState<boolean>(
    () => sessionStorage.getItem(`ts-edit-mode-${projectId}`) === "true",
  );
  const [ongoingAction, setOngoingAction] = useState(false);

  // Viewers can't enter edit mode - gated here (rather than only disabling the
  // toolbar button) so the "e" keyboard shortcut is blocked too.
  const setEditMode = useCallback(
    (update: boolean | ((prev: boolean) => boolean)) => {
      setEditModeRaw((prev) => {
        const next = typeof update === "function" ? update(prev) : update;
        return canAnnotate ? next : false;
      });
    },
    [canAnnotate],
  );

  // If the role check resolves to "can't annotate" after edit mode was already on
  // (e.g. restored from a previous session, or a mid-session role change), drop back
  // to view mode.
  useEffect(() => {
    if (!canAnnotate) setEditModeRaw(false);
  }, [canAnnotate]);

  // Persist editMode to sessionStorage on every change
  useEffect(() => {
    if (!projectId) return;
    sessionStorage.setItem(`ts-edit-mode-${projectId}`, String(editMode));
  }, [editMode, projectId]);

  // Persist activeTool to sessionStorage on every change. Skipped while a
  // restore is pending, so the initial null does not wipe the saved tool.
  useEffect(() => {
    if (!projectId || pendingTool) return;
    if (activeTool) {
      sessionStorage.setItem(
        activeToolKey(projectId),
        JSON.stringify(activeTool),
      );
    } else {
      sessionStorage.removeItem(activeToolKey(projectId));
    }
  }, [activeTool, projectId, pendingTool]);

  const syncTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastSyncCount = useRef<number>(0);

  const parseRawAnnotations = useCallback(
    (annotations: Annotation[]): TimeSeriesAnnotation[] => {
      const parsedAnnotations: TimeSeriesAnnotation[] = [];
      annotations.forEach((annotation) => {
        const parsedAnnotation = convertRawAnnotationsToTimeSeries(annotation);
        if (parsedAnnotation) parsedAnnotations.push(parsedAnnotation);
      });
      return parsedAnnotations;
    },
    [],
  );

  const parseTimeSeriesAnnotations = useCallback(
    (annotations: TimeSeriesAnnotation[]): Annotation[] => {
      const parsedAnnotations: Annotation[] = [];
      annotations.forEach((annotation) => {
        const parsedAnnotation = convertTimeSeriesToRawAnnotations(annotation);
        if (parsedAnnotation) parsedAnnotations.push(parsedAnnotation);
      });
      return parsedAnnotations;
    },
    [],
  );

  // The sample's annotations are shared with other tools, so this view must replace
  // only its own and carry the rest through untouched. Without this, annotations it
  // cannot represent - shot labels, for example - are lost on every edit.
  const mergeTimeSeriesAnnotations = useCallback(
    (previous: Annotation[], updated: TimeSeriesAnnotation[]): Annotation[] => [
      ...previous.filter((annotation) => !isTimeSeriesAnnotation(annotation)),
      ...parseTimeSeriesAnnotations(updated),
    ],
    [parseTimeSeriesAnnotations],
  );

  // Discards any in-progress annotation for the currently active tool and clears the
  // ongoing-action flag - used whenever a draw is abandoned rather than completed normally
  const cancelOngoingAction = useCallback(() => {
    if (!ongoingAction) return;
    if (activeTool) {
      toolingCallbacks.get(activeTool.type)?.cancel?.();
    }
    setOngoingAction(false);
  }, [activeTool, ongoingAction, toolingCallbacks]);

  useEffect(() => {
    if (!project) return;
    const timeSeriesCategories: Map<string, TimeSeriesCategory> = new Map();
    if (project.time_point_labels) {
      project.time_point_labels.forEach((label, index) => {
        const category_id = `${TimeSeriesAnnotationType.TIME_POINT}_${label}`;
        timeSeriesCategories.set(category_id, {
          label,
          color: randomColor(index),
          type: TimeSeriesAnnotationType.TIME_POINT,
        });
      });
    }
    if (project.time_region_labels) {
      project.time_region_labels.forEach((label, index) => {
        const category_id = `${TimeSeriesAnnotationType.TIME_REGION}_${label}`;
        timeSeriesCategories.set(category_id, {
          label,
          color: randomColor(index),
          type: TimeSeriesAnnotationType.TIME_REGION,
        });
      });
    }
    if (project.bounding_box_labels) {
      project.bounding_box_labels.forEach((label, index) => {
        const category_id = `${TimeSeriesAnnotationType.BOUNDING_BOX}_${label}`;
        timeSeriesCategories.set(category_id, {
          label,
          color: randomColor(index),
          type: TimeSeriesAnnotationType.BOUNDING_BOX,
        });
      });
    }
    if (project.polygon_labels) {
      project.polygon_labels.forEach((label, index) => {
        const category_id = `${TimeSeriesAnnotationType.POLYGON}_${label}`;
        timeSeriesCategories.set(category_id, {
          label,
          color: randomColor(index),
          type: TimeSeriesAnnotationType.POLYGON,
        });
      });
    }
    setCategories(timeSeriesCategories);
  }, [project]);

  // This is a reference to allow the up-to-date function to be called from within an effect without triggering a refresh
  const cancelOngoingActionRef = useRef(cancelOngoingAction);
  useEffect(() => {
    cancelOngoingActionRef.current = cancelOngoingAction;
  }, [cancelOngoingAction]);

  useEffect(() => {
    cancelOngoingActionRef.current(); // If the annotations are changed, any ongoing annotations must be cancelled
    setAnnotations(parseRawAnnotations(rawAnnotations));
  }, [parseRawAnnotations, rawAnnotations]);

  const triggerSync = useCallback(() => {
    setSyncCounter((prev) => (prev + 1) % 100);
  }, []);

  const syncAnnotations = useCallback(() => {
    if (ongoingAction) {
      if (syncTimeoutRef.current !== null) {
        clearTimeout(syncTimeoutRef.current);
      }
      syncTimeoutRef.current = setTimeout(triggerSync, 100);
      return;
    }
    syncTimeoutRef.current = null;
    setRawAnnotations((prev) => mergeTimeSeriesAnnotations(prev, annotations));
  }, [
    annotations,
    ongoingAction,
    mergeTimeSeriesAnnotations,
    setRawAnnotations,
    triggerSync,
  ]);

  useEffect(() => {
    if (lastSyncCount.current === syncCounter) return;
    lastSyncCount.current = syncCounter;
    syncAnnotations();
  }, [syncAnnotations, syncCounter]);

  // A new annotation belongs to whoever is drawing it, so it carries their username
  // from the moment it appears in the table - the same value the server stamps on it
  // when it is saved. "manual" is only a fallback for the brief window before the
  // auth context resolves.
  const createAnnotation = useCallback(
    (type: TimeSeriesAnnotationType, label: string): TimeSeriesAnnotation => {
      const id = uuidv4();
      return {
        id,
        db_id: null,
        created_by: user?.username ?? "manual",
        label,
        signal_name: signalName,
        type,
        points: [],
        selected: false,
      };
    },
    [signalName, user],
  );

  const addAnnotation = useCallback(
    (annotation: TimeSeriesAnnotation) => {
      if (syncTimeoutRef.current !== null) {
        clearTimeout(syncTimeoutRef.current);
        syncTimeoutRef.current = null;
      }
      syncTimeoutRef.current = setTimeout(triggerSync, 100);

      setAnnotations((prev) => [...prev, annotation]);
    },
    [triggerSync],
  );

  const removeAnnotation = useCallback(
    (id: string) => {
      if (syncTimeoutRef.current !== null) {
        clearTimeout(syncTimeoutRef.current);
      }
      syncTimeoutRef.current = setTimeout(triggerSync, 100);

      setAnnotations((prev) =>
        prev.filter((annotation) => annotation.id !== id),
      );
    },
    [triggerSync],
  );

  const getAnnotation = useCallback(
    (id: string) => {
      annotations.forEach((annotation) => {
        if (annotation.id === id) return annotation;
      });
      console.warn(`Annotation with id: ${id} could not be found`);
      return null;
    },
    [annotations],
  );

  const registerTooling = useCallback(
    (type: TimeSeriesAnnotationType, callbacks: ToolingCallbacks) => {
      setToolingCallbacks((prev) => {
        if (prev.has(type)) return prev;
        const newMap = new Map(prev);
        newMap.set(type, callbacks);
        return newMap;
      });
    },
    [],
  );

  const setAnnotationTool = useCallback(
    (tool: TimeSeriesToolDefinition | null) => {
      if (!tool || toolingCallbacks.has(tool.type)) {
        cancelOngoingAction(); // Switching tools mid-draw abandons whatever was in progress
        setActiveTool(tool);
        return;
      }
      console.warn(
        `Could not set ${tool.type} as active tool since no callback has been registered`,
      );
    },
    [cancelOngoingAction, toolingCallbacks],
  );

  // Lets a mid-draw annotation be abandoned via Escape, in addition to switching tools
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cancelOngoingAction();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cancelOngoingAction]);

  // Apply a restored tool once tooling and categories have registered. The
  // label is checked against the project's current categories, since it may
  // have been removed since the tool was saved; setAnnotationTool applies the
  // remaining guard that a callback exists for the type.
  useEffect(() => {
    if (!pendingTool) return;
    if (toolingCallbacks.size === 0 || categories.size === 0) return;

    const labelExists = Array.from(categories.values()).some(
      (category) =>
        category.type === pendingTool.type &&
        category.label === pendingTool.label,
    );
    if (labelExists) {
      setAnnotationTool(pendingTool);
    } else {
      sessionStorage.removeItem(activeToolKey(projectId));
    }
    setPendingTool(null);
  }, [pendingTool, toolingCallbacks, categories, projectId, setAnnotationTool]);

  const updateAnnotation = useCallback(
    (annotation: TimeSeriesAnnotation) => {
      if (syncTimeoutRef.current !== null) {
        clearTimeout(syncTimeoutRef.current);
      }
      syncTimeoutRef.current = setTimeout(triggerSync, 100);

      setAnnotations((prev) =>
        prev.map((item) =>
          item.id === annotation.id
            ? { ...annotation, selected: item.selected }
            : item,
        ),
      );
    },
    [triggerSync],
  );

  const triggerUpdate = useCallback(() => {
    setUpdateCounter((prev) => (prev + 1) % 100);
  }, []);

  const selectAnnotations = useCallback(
    (ids: string[]) => {
      if (!editMode) return;

      const updated_state: TimeSeriesAnnotation[] = annotations.map(
        (annotation) => {
          if (ids.includes(annotation.id)) {
            return { ...annotation, selected: true };
          }
          return { ...annotation, selected: false };
        },
      );

      setAnnotations(updated_state);
    },
    [annotations, editMode],
  );

  const findSelectedAnnotations = useCallback(
    (range: SelectionRange | null) => {
      if (!editMode) return;

      if (!range) {
        const updated_state: TimeSeriesAnnotation[] = annotations.map(
          (annotation) => ({ ...annotation, selected: false }),
        );
        setAnnotations(updated_state);
        return;
      }

      const updated_state: TimeSeriesAnnotation[] = annotations.map(
        (annotation) => {
          if (annotation.type === TimeSeriesAnnotationType.TIME_REGION) {
            if (
              annotation.points[0].x > range.x.low &&
              annotation.points[1].x < range.x.high
            ) {
              return { ...annotation, selected: true };
            }
            return { ...annotation, selected: false };
          }
          if (annotation.type === TimeSeriesAnnotationType.TIME_POINT) {
            if (
              annotation.points[0].x > range.x.low &&
              annotation.points[0].x < range.x.high
            ) {
              return { ...annotation, selected: true };
            }
            return { ...annotation, selected: false };
          }
          if (annotation.type === TimeSeriesAnnotationType.BOUNDING_BOX) {
            if (
              annotation.points[0].x > range.x.low &&
              annotation.points[1].x < range.x.high &&
              annotation.points[1].y > range.y.low &&
              annotation.points[0].y < range.y.high
            ) {
              return { ...annotation, selected: true };
            }
            return { ...annotation, selected: false };
          }
          if (annotation.type === TimeSeriesAnnotationType.POLYGON) {
            const selected =
              annotation.points.length > 0 &&
              annotation.points.every(
                (point) =>
                  point.x >= range.x.low &&
                  point.x <= range.x.high &&
                  point.y >= range.y.low &&
                  point.y <= range.y.high,
              );

            return {
              ...annotation,
              selected,
            };
          }
          return { ...annotation, selected: false };
        },
      );

      setAnnotations(updated_state);
    },
    [annotations, editMode],
  );

  const batchUpdateLabels = useCallback(
    (category: TimeSeriesCategory) => {
      const updatedState: TimeSeriesAnnotation[] = annotations.map(
        (annotation) => {
          // Label should only be changed if it is the annotation is the correct type and selected
          if (annotation.type === category.type && annotation.selected) {
            return { ...annotation, label: category.label };
          }
          return annotation;
        },
      );

      setRawAnnotations((prev) =>
        mergeTimeSeriesAnnotations(prev, updatedState),
      );
    },
    [annotations, mergeTimeSeriesAnnotations, setRawAnnotations],
  );

  // Writes straight through to the sample's annotations, unlike removeAnnotation,
  // which only drops a half-drawn shape from this view's own working copy.
  const deleteAnnotations = useCallback(
    (shouldDelete: (annotation: TimeSeriesAnnotation) => boolean) => {
      setRawAnnotations((prev) =>
        mergeTimeSeriesAnnotations(
          prev,
          annotations.filter((annotation) => !shouldDelete(annotation)),
        ),
      );
    },
    [annotations, mergeTimeSeriesAnnotations, setRawAnnotations],
  );

  const batchDeleteAnnotations = useCallback(
    () => deleteAnnotations((annotation) => annotation.selected ?? false),
    [deleteAnnotations],
  );

  const actionsValue: TimeSeriesActions = useMemo(
    () => ({
      setAnnotations,
      createAnnotation,
      addAnnotation,
      removeAnnotation,
      setAnnotationTool,
      registerTooling,
      updateAnnotation,
      getAnnotation,
      triggerUpdate,
      selectAnnotations,
      findSelectedAnnotations,
      setEditMode,
      setOngoingAction,
    }),
    [
      createAnnotation,
      addAnnotation,
      removeAnnotation,
      setAnnotationTool,
      registerTooling,
      updateAnnotation,
      getAnnotation,
      triggerUpdate,
      selectAnnotations,
      findSelectedAnnotations,
      setEditMode,
    ],
  );

  const stateValue: TimeSeriesState = useMemo(
    () => ({
      annotations,
      activeAnnotationTool: activeTool,
      toolingCallbacks,
      forceUpdate: updateCounter,
      isDrawing,
      ongoingAction,
      categories,
      editMode,
      canAnnotate,
    }),
    [
      annotations,
      activeTool,
      toolingCallbacks,
      updateCounter,
      isDrawing,
      ongoingAction,
      categories,
      editMode,
      canAnnotate,
    ],
  );

  useEffect(() => {
    if (!editMode) return;

    const deleteSelection = (event: KeyboardEvent) => {
      if (event.key === "Delete" || event.key === "Backspace") {
        batchDeleteAnnotations();
      }
    };

    document.addEventListener("keydown", deleteSelection);

    return () => {
      document.removeEventListener("keydown", deleteSelection);
    };
  }, [
    annotations,
    batchDeleteAnnotations,
    editMode,
    parseTimeSeriesAnnotations,
    setRawAnnotations,
  ]);

  // setEditMode's identity changes whenever canAnnotate resolves (see above), so it's
  // read through a ref rather than a dependency here - re-registering these listeners
  // on that change would tear down the "Control" keydown/keyup pair mid-drag and break
  // the ctrl-drag gesture tools rely on to draw. The ref always has the latest guard.
  const setEditModeRef = useRef(setEditMode);
  useEffect(() => {
    setEditModeRef.current = setEditMode;
  }, [setEditMode]);

  useEffect(() => {
    const keyDownHandler = (event: KeyboardEvent) => {
      if (isEditableEventTarget(event.target)) return;

      if (event.key === "Control") {
        setIsDrawing(true);
      }

      if (event.key === "e") {
        setEditModeRef.current((prev) => !prev);
      }
    };

    const keyUpHandler = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsDrawing(false);
      }
    };

    document.addEventListener("keydown", keyDownHandler);
    document.addEventListener("keyup", keyUpHandler);

    return () => {
      document.removeEventListener("keydown", keyDownHandler);
      document.removeEventListener("keyup", keyUpHandler);
    };
  }, []);

  const annotationLabels = Array.from(categories.values()).map(
    (category, index) => {
      return (
        <Item
          key={`update${index}`}
          id={`update${index}`}
          hidden={({ props }) => props.annotation.type !== category.type}
          onClick={({ props }) => {
            const annotation = props.annotation as TimeSeriesAnnotation;
            // If this annotation is selected, batch update all selected annotation
            if (annotation.selected) {
              batchUpdateLabels(category);
              return;
            }

            // If the annotation is not selected, only update this one
            const newAnnotation: TimeSeriesAnnotation = {
              ...props.annotation,
              label: category.label,
            };
            updateAnnotation(newAnnotation);
          }}
        >
          {category.label}
        </Item>
      );
    },
  );

  return (
    <TimeSeriesActionsContext.Provider value={actionsValue}>
      <TimeSeriesStateContext value={stateValue}>
        {children}
        <Menu id={`${TIME_SERIES_ANNOTATION_MENU}`}>
          <Item
            id="delete"
            onClick={({ props }: ItemParams) => {
              const annotation = props.annotation as TimeSeriesAnnotation;
              // If this annotation is selected, batch delete all selected annotation
              if (annotation.selected) {
                batchDeleteAnnotations();
                return;
              }

              // If the annotation is not selected, only delete this one
              deleteAnnotations(
                (candidate) => candidate.id === props.annotation.id,
              );
            }}
          >
            Delete
          </Item>
          <Submenu label="Set type">{annotationLabels}</Submenu>
        </Menu>
      </TimeSeriesStateContext>
    </TimeSeriesActionsContext.Provider>
  );
};
