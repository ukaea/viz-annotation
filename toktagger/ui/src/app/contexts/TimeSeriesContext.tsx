"use client";

import {
  Annotation,
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
import {
  convertRawAnnotationsToTimeSeries,
  convertTimeSeriesToRawAnnotations,
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
  findSelectedAnnotations: (
    range: { low: number; high: number } | null,
  ) => void;
  setEditMode: (turnOn: boolean) => void;
  setOngoingAction: (state: boolean) => void;
};

type TimeSeriesState = {
  annotations: TimeSeriesAnnotation[];
  activeAnnotationTool: TimeSeriesToolDefinition | null;
  toolingCallbacks: Map<TimeSeriesAnnotationType, ToolingCallbacks>;
  forceUpdate: number;
  isDrawing: boolean;
  categories: Map<string, TimeSeriesCategory>;
  editMode: boolean;
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

export const TimeSeriesProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const {
    annotations: rawAnnotations,
    setAnnotations: setRawAnnotations,
    project,
  } = useSample();

  const [annotations, setAnnotations] = useState<TimeSeriesAnnotation[]>([]);
  const [toolingCallbacks, setToolingCallbacks] = useState<
    Map<TimeSeriesAnnotationType, ToolingCallbacks>
  >(new Map());
  const [activeTool, setActiveTool] = useState<TimeSeriesToolDefinition | null>(
    null,
  );
  const [updateCounter, setUpdateCounter] = useState(0);
  const [syncCounter, setSyncCounter] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [categories, setCategories] = useState<Map<string, TimeSeriesCategory>>(
    new Map(),
  );
  const [editMode, setEditMode] = useState(false);
  const [ongoingAction, setOngoingAction] = useState(false);

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
    setCategories(timeSeriesCategories);
  }, [project]);

  useEffect(() => {
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
    const rawAnnotations = parseTimeSeriesAnnotations(annotations);
    setRawAnnotations((_prev) => rawAnnotations);
  }, [
    annotations,
    ongoingAction,
    parseTimeSeriesAnnotations,
    setRawAnnotations,
    triggerSync,
  ]);

  useEffect(() => {
    if (lastSyncCount.current === syncCounter) return;
    lastSyncCount.current = syncCounter;
    syncAnnotations();
  }, [syncAnnotations, syncCounter]);

  const createAnnotation = useCallback(
    (type: TimeSeriesAnnotationType, label: string): TimeSeriesAnnotation => {
      const id = uuidv4();
      return {
        id,
        created_by: "manual",
        label,
        type,
        points: [],
        selected: false,
      };
    },
    [],
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
      const currentAnnotations = annotations;
      const newAnnotations = currentAnnotations.filter(
        (annotation) => annotation.id !== id,
      );
      setRawAnnotations((_prev) => parseTimeSeriesAnnotations(newAnnotations));
    },
    [annotations, parseTimeSeriesAnnotations, setRawAnnotations],
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
        setActiveTool(tool);
        return;
      }
      console.warn(
        `Could not set ${tool.type} as active tool since no callback has been registered`,
      );
    },
    [toolingCallbacks],
  );

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
    (range: { low: number; high: number } | null) => {
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
              annotation.points[0].x > range.low &&
              annotation.points[1].x < range.high
            ) {
              return { ...annotation, selected: true };
            }
            return { ...annotation, selected: false };
          }
          if (annotation.type === TimeSeriesAnnotationType.TIME_POINT) {
            if (
              annotation.points[0].x > range.low &&
              annotation.points[0].x < range.high
            ) {
              return { ...annotation, selected: true };
            }
            return { ...annotation, selected: false };
          }
          if (annotation.type === TimeSeriesAnnotationType.BOUNDING_BOX) {
            if (
              annotation.points[0].x > range.low &&
              annotation.points[0].x < range.high
            ) {
              return { ...annotation, selected: true };
            }
            return { ...annotation, selected: false };
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
      const updated_state: TimeSeriesAnnotation[] = annotations.map(
        (annotation) => {
          // Label should only be changed if it is the annotation is the correct type and selected
          if (annotation.type === category.type && annotation.selected) {
            return { ...annotation, label: category.label };
          }
          return annotation;
        },
      );

      setAnnotations(updated_state);
    },
    [annotations],
  );

  const batchDeleteAnnotations = useCallback(() => {
    const updatedState = annotations.filter(
      (annotation) => !annotation.selected,
    );
    setRawAnnotations((_prev) => parseTimeSeriesAnnotations(updatedState));
  }, [annotations, parseTimeSeriesAnnotations, setRawAnnotations]);

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
    ],
  );

  const stateValue: TimeSeriesState = useMemo(
    () => ({
      annotations,
      activeAnnotationTool: activeTool,
      toolingCallbacks,
      forceUpdate: updateCounter,
      isDrawing,
      categories,
      editMode,
    }),
    [
      annotations,
      activeTool,
      toolingCallbacks,
      updateCounter,
      isDrawing,
      categories,
      editMode,
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

  useEffect(() => {
    const keyDownHandler = (event: KeyboardEvent) => {
      if (event.key === "Control") {
        setIsDrawing(true);
      }

      if (event.key === "e") {
        setEditMode((prev) => !prev);
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
  }, [editMode]);

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
              removeAnnotation(props.annotation.id);
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
