"use client";
import { Project, type NavAdapter } from "@/types";
import {
  Flex,
  Button,
  ActionButton,
  ButtonGroup,
  ToastQueue,
  Text,
  View,
  Checkbox,
  Tooltip,
  TooltipTrigger,
  SearchField,
} from "@adobe/react-spectrum";
import { useCallback, useEffect, useState } from "react";
import StepForward from "@spectrum-icons/workflow/StepForward";
import StepBackward from "@spectrum-icons/workflow/StepBackward";
import SaveFloppy from "@spectrum-icons/workflow/SaveFloppy";
import Delete from "@spectrum-icons/workflow/Delete";
import { getShotSample, saveSampleAnnotations, updateSample, BACKEND_API_URL, apiFetch, getAnnotationsForSample } from "@/app/core";
import { useAuth } from "@/app/contexts/AuthContext";
import {
  useNavigate,
  NavigateFunction,
  useSearchParams,
} from "react-router-dom";
import { useSample } from "@/app/contexts/SampleContext";
import { useSampleHistory } from "@/app/contexts/SampleHistoryContext";
import { getNextSample } from "@/app/core";
import type { SortDescriptor, SortDirection, Key } from "@react-types/shared";
import { useNavAdapter } from "@/app/contexts/NavAdapterContext";

const TOAST_TIMEOUT = 5000;

async function navigateToSample(
  project_id: string,
  sample_id: string,
  navigate: NavigateFunction,
  sortDescriptor: SortDescriptor | null,
) {
  const params = new URLSearchParams();
  if (sortDescriptor) {
    params.append("sortColumn", sortDescriptor.column.toString());
    params.append("sortDirection", sortDescriptor.direction);
  }

  try {
    const NEXT_SAMPLE_URL =
      `/ui/projects/${project_id}/samples/${sample_id}` +
      (params.toString() ? `?${params.toString()}` : "");
    navigate(NEXT_SAMPLE_URL);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    ToastQueue.negative(`Failed to fetch next sample: ${message}`, {
      timeout: TOAST_TIMEOUT,
    });
  }
}

async function navigateToNextSample(
  project_id: string,
  navigate: NavigateFunction,
  visited_sample_ids: string[],
  sortDescriptor: SortDescriptor | null,
) {
  const sample = await getNextSample(
    project_id,
    visited_sample_ids,
    sortDescriptor,
  );
  if (!sample) {
    ToastQueue.negative("No more samples available!", {
      timeout: TOAST_TIMEOUT,
    });
    return;
  }
  navigateToSample(project_id, sample._id, navigate, sortDescriptor);
}

type ButtonInfo = {
  project_id: string;
  sample_id: string;
  setIsValidated: (validated: boolean) => void;
  navAdapter: NavAdapter;
};

type SaveButtonInfo = ButtonInfo & {
  saveOnNavigate?: boolean;
};

type NextButtonInfo = ButtonInfo & {
  saveOnNavigate?: boolean;
  visitedSampleIds: string[];
  sortDescriptor: SortDescriptor | null;
};

type PreviousButtonInfo = ButtonInfo & {
  saveOnNavigate?: boolean;
  isDisabled: boolean;
  sortDescriptor: SortDescriptor | null;
  popVisitedSampleId: () => string | null;
};

function NextButton({
  project_id,
  sample_id,
  setIsValidated,
  visitedSampleIds,
  sortDescriptor,
  saveOnNavigate,
  navAdapter,
}: NextButtonInfo) {
  const navigate = useNavigate();

  const moveNextShot = useCallback(async () => {
    const annotationsToSave = navAdapter.getAnnotations();
    await saveSampleAnnotations(
      project_id,
      sample_id,
      annotationsToSave,
      saveOnNavigate,
    );
    if (saveOnNavigate) {
      navAdapter.afterSave?.();
      setIsValidated(true);
    }
    await navigateToNextSample(
      project_id,
      navigate,
      visitedSampleIds,
      sortDescriptor,
    );
  }, [
    project_id,
    sample_id,
    navigate,
    saveOnNavigate,
    setIsValidated,
    visitedSampleIds,
    sortDescriptor,
    navAdapter,
  ]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Check for Shift + Right Arrow
      if (e.shiftKey && e.key === "ArrowRight") {
        e.preventDefault();
        moveNextShot();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sample_id, project_id, navigate, moveNextShot]);

  return (
    <View marginStart="size-100">
      <ActionButton aria-label="Next Sample" onPress={moveNextShot}>
        <StepForward />
      </ActionButton>
    </View>
  );
}

export function JumpToNextButton({
  project,
  sortDescriptor,
}: {
  project: Project;
  sortDescriptor: SortDescriptor;
}) {
  const navigate = useNavigate();

  const moveNextShot = useCallback(async () => {
    if (project._id) {
      await navigateToNextSample(project._id, navigate, [], sortDescriptor);
    }
  }, [project._id, navigate, sortDescriptor]);

  return (
    <View marginStart="size-100">
      <Button
        variant="primary"
        aria-label="Jump to Next Sample"
        onPress={moveNextShot}
      >
        <Text>Jump to Next Sample</Text> <StepForward />
      </Button>
    </View>
  );
}

function PreviousButton({
  project_id,
  sample_id,
  setIsValidated,
  isDisabled,
  popVisitedSampleId,
  saveOnNavigate,
  sortDescriptor,
  navAdapter,
}: PreviousButtonInfo) {
  const navigate = useNavigate();

  const movePreviousShot = useCallback(async () => {
    const annotationsToSave = navAdapter.getAnnotations();
    await saveSampleAnnotations(
      project_id,
      sample_id,
      annotationsToSave,
      saveOnNavigate,
    );
    if (saveOnNavigate) {
      navAdapter.afterSave?.();
      setIsValidated(true);
    }

    const previous_sample_id: string | null = popVisitedSampleId();

    if (!previous_sample_id) {
      ToastQueue.negative("No earlier samples available!", {
        timeout: TOAST_TIMEOUT,
      });
      return;
    }
    navigateToSample(project_id, previous_sample_id, navigate, sortDescriptor);
  }, [
    project_id,
    sample_id,
    navigate,
    saveOnNavigate,
    popVisitedSampleId,
    sortDescriptor,
    setIsValidated,
    navAdapter,
  ]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Check for Shift + Left Arrow
      if (e.shiftKey && e.key === "ArrowLeft") {
        e.preventDefault();
        movePreviousShot();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sample_id, project_id, navigate, movePreviousShot]);

  return (
    <View marginStart="size-100">
      <ActionButton
        isDisabled={isDisabled}
        aria-label="Previous Sample"
        onPress={movePreviousShot}
      >
        <StepBackward />
      </ActionButton>
    </View>
  );
}

function SaveButton({
  project_id,
  sample_id,
  setIsValidated,
  saveOnNavigate: _saveOnNavigate,
  navAdapter,
}: SaveButtonInfo) {
  const handleClick = async () => {
    try {
      const annotationsToSave = navAdapter.getAnnotations();
      await saveSampleAnnotations(
        project_id,
        sample_id,
        annotationsToSave,
        true,
      );
      navAdapter.afterSave?.();
      ToastQueue.positive(`Saved ${annotationsToSave.length} annotations!`, {
        timeout: TOAST_TIMEOUT,
      });
      setIsValidated(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      ToastQueue.negative(`Failed to save annotations: ${message}`, {
        timeout: TOAST_TIMEOUT,
      });
    }
  };

  return (
    <View marginStart="size-100">
      <ActionButton aria-label="Save" onPress={handleClick}>
        <SaveFloppy />
        <Text>Save</Text>
      </ActionButton>
    </View>
  );
}

function ClearButton({
  project_id,
  sample_id,
  setIsValidated,
  navAdapter,
}: ButtonInfo) {
  const handleClick = () => {
    navAdapter.clear();
    // Mark as unvalidated annotations
    updateSample(project_id, sample_id, { validated_annotations: false });
    setIsValidated(false);
  };

  return (
    <View marginStart="size-100">
      <ActionButton aria-label="Clear" onPress={handleClick}>
        <Delete />
        <Text>Clear</Text>
      </ActionButton>
    </View>
  );
}

type SaveInfo = {
  project_id: string;
  sample_id: string;
  sortDescriptor: SortDescriptor | null;
  saveOnNavigate?: boolean;
  setIsValidated: (validated: boolean) => void;
  navAdapter: NavAdapter;
};

export function ShotSearch({
  project_id,
  sample_id,
  sortDescriptor,
  saveOnNavigate,
  setIsValidated,
  navAdapter,
}: SaveInfo) {
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = useState<string>("");

  const onSearchSubmit = async (newValue: string) => {
    if (newValue == "") {
      setErrorMessage("");
    } else if (/^[0-9]*$/.test(newValue)) {
      setErrorMessage("");
      const shot_id = newValue;
      try {
        const sample = await getShotSample(project_id, shot_id);
        if (sample !== null) {
          const annotationsToSave = navAdapter.getAnnotations();
          await saveSampleAnnotations(
            project_id,
            sample_id,
            annotationsToSave,
            saveOnNavigate,
          );
          if (saveOnNavigate) {
            navAdapter.afterSave?.();
            setIsValidated(true);
          }
          navigateToSample(project_id, sample._id, navigate, sortDescriptor);
        } else {
          setErrorMessage("Shot not found!");
        }
      } catch (err) {
        console.error("Failed to fetch data:", err);
      }
    } else {
      setErrorMessage("Please enter a number.");
    }
  };

  return (
    <SearchField
      label="Jump to Shot"
      onSubmit={onSearchSubmit}
      validationState={errorMessage ? "invalid" : undefined}
      errorMessage={errorMessage}
    ></SearchField>
  );
}

type NavigationBarInfo = {
  project_id: string;
  sample_id: string;
};
export function NavigationBar({ project_id, sample_id }: NavigationBarInfo) {
  const { setIsValidated, setAnnotations } = useSample();
  const { user } = useAuth();
  const navAdapter = useNavAdapter();

  const {
    visitedSampleIds,
    popVisitedSampleId,
    SaveOnNavigate,
    setSaveOnNavigate,
  } = useSampleHistory();

  const [searchParamsObj] = useSearchParams();
  const [sortDescriptor] = useState<SortDescriptor | null>(() => {
    const column: Key | null = searchParamsObj.get("sortColumn");
    const raw_direction: string | null = searchParamsObj.get("sortDirection");
    if (!column || !raw_direction) {
      return null;
    }
    const direction: SortDirection =
      (raw_direction as SortDirection) || "ascending";
    return { column, direction };
  });

  const [showOthers, setShowOthers] = useState(true);

  const toggleShowOthers = useCallback(async (next: boolean) => {
    setShowOthers(next);
    if (user) {
      await apiFetch(
        `${BACKEND_API_URL}/projects/${project_id}/members/${user._id}`,
        { method: "PUT", body: JSON.stringify({ show_others_annotations: next }) },
      );
    }
    // Re-fetch annotations with updated visibility
    const fresh = await getAnnotationsForSample(project_id, sample_id);
    setAnnotations(() => fresh);
  }, [project_id, sample_id, user, setAnnotations]);

  return (
    <Flex alignItems="center" direction="column" gap="size-100">
      <ButtonGroup>
        <SaveButton
          project_id={project_id}
          sample_id={sample_id}
          setIsValidated={setIsValidated}
          navAdapter={navAdapter}
        />
        <PreviousButton
          project_id={project_id}
          sample_id={sample_id}
          setIsValidated={setIsValidated}
          isDisabled={visitedSampleIds.length == 1}
          popVisitedSampleId={popVisitedSampleId}
          saveOnNavigate={SaveOnNavigate}
          sortDescriptor={sortDescriptor}
          navAdapter={navAdapter}
        />
        <NextButton
          project_id={project_id}
          sample_id={sample_id}
          setIsValidated={setIsValidated}
          visitedSampleIds={visitedSampleIds}
          saveOnNavigate={SaveOnNavigate}
          sortDescriptor={sortDescriptor}
          navAdapter={navAdapter}
        />
        <ClearButton
          project_id={project_id}
          sample_id={sample_id}
          setIsValidated={setIsValidated}
          navAdapter={navAdapter}
        />
      </ButtonGroup>
      <TooltipTrigger delay={1000} placement="bottom">
        <Checkbox isSelected={SaveOnNavigate} onChange={setSaveOnNavigate}>
          Save on Navigate
        </Checkbox>
        <Tooltip>
          When enabled, annotations will be saved when navigating to another
          sample.
        </Tooltip>
      </TooltipTrigger>
      <TooltipTrigger delay={1000} placement="bottom">
        <Checkbox isSelected={showOthers} onChange={toggleShowOthers}>
          Show Others&apos; Annotations
        </Checkbox>
        <Tooltip>
          When enabled, annotations from other users are also displayed.
        </Tooltip>
      </TooltipTrigger>
      <ShotSearch
        project_id={project_id}
        sample_id={sample_id}
        sortDescriptor={sortDescriptor}
        saveOnNavigate={SaveOnNavigate}
        setIsValidated={setIsValidated}
        navAdapter={navAdapter}
      />
    </Flex>
  );
}
