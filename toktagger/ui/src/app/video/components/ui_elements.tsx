"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  ActionButton,
  Button,
  ComboBox,
  Divider,
  Flex,
  Item,
  Tooltip,
  TooltipTrigger,
  ToggleButton,
  useProvider,
  View,
} from "@adobe/react-spectrum";
import StepBackward from "@spectrum-icons/workflow/StepBackward";
import Draw from "@spectrum-icons/workflow/Draw";
import ImageMapPolygon from "@spectrum-icons/workflow/ImageMapPolygon";
import ImageMapRectangle from "@spectrum-icons/workflow/ImageMapRectangle";
import FullScreenExit from "@spectrum-icons/workflow/FullScreenExit";
import Target from "@spectrum-icons/workflow/Target";

/**
 * Minimal class category shape consumed by the class picker.
 * Kept intentionally small so this file stays UI-only and decoupled from session/types.
 */
export type ClassCategoryItem = {
  name: string;
};

/**
 * Class selector used to "arm" drawing with a label.
 * When no class is selected, the annotator should disable drawing.
 */
export function ClassPanel({
  items,
  selectedClassName,
  setSelectedClassName,
}: {
  items: Iterable<ClassCategoryItem>;
  selectedClassName: string | null;
  setSelectedClassName: (v: string | null) => void;
}) {
  return (
    <View marginX="auto" width="12rem">
      <ComboBox
        label={
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">
            Class Label
          </span>
        }
        items={items}
        selectedKey={selectedClassName}
        onSelectionChange={(key) =>
          setSelectedClassName((key as string) || null)
        }
        width="100%"
      >
        {(item) => <Item key={item.name}>{item.name}</Item>}
      </ComboBox>
    </View>
  );
}

/**
 * Center frame control:
 * - Default state: "Frame X" button
 * - Edit state: inline numeric input in the same pill footprint
 */
export function FrameJumpField(props: {
  frame: number;
  onJump: (n: number) => void;
}) {
  const { colorScheme } = useProvider();
  const isDark = colorScheme === "dark";
  const [isEditing, setIsEditing] = useState(false);
  const [draftValue, setDraftValue] = useState<string>(String(props.frame));
  const [pillWidth, setPillWidth] = useState<number | null>(null);
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const displayPillRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isEditing) {
      setDraftValue(String(props.frame));
    }
  }, [props.frame, isEditing]);

  useEffect(() => {
    if (!isEditing) return;

    const rafId = requestAnimationFrame(() => {
      const input = inputRef.current;
      if (!input) return;
      input.focus();
      input.select();
    });

    return () => cancelAnimationFrame(rafId);
  }, [isEditing]);

  const startEdit = () => {
    const width = displayPillRef.current?.offsetWidth ?? null;
    setPillWidth(width);
    setDraftValue(String(props.frame));
    setIsEditing(true);
  };

  const cancelEdit = () => {
    setDraftValue(String(props.frame));
    setIsFocused(false);
    setIsEditing(false);
  };

  const commitEdit = () => {
    const trimmed = draftValue.trim();
    if (trimmed === "") {
      cancelEdit();
      return;
    }

    const parsed = Number(trimmed);
    if (Number.isInteger(parsed) && parsed >= 0) {
      setIsEditing(false);
      props.onJump(Math.trunc(parsed));
    }
  };

  if (!isEditing) {
    return (
      <div ref={displayPillRef} style={{ display: "inline-flex" }}>
        <Button variant="primary" onPress={startEdit}>
          Frame {props.frame}
        </Button>
      </div>
    );
  }

  return (
    <View
      role="presentation"
      borderWidth="thin"
      borderColor={isDark ? "static-white" : "gray-900"}
      backgroundColor={isDark ? "transparent" : "static-white"}
      height={32}
      paddingX={12}
      width={pillWidth !== null ? pillWidth : undefined}
      UNSAFE_style={{
        borderRadius: "9999px",
        boxShadow: isFocused
          ? `0 0 0 2px ${isDark ? "rgba(255, 255, 255, 0.7)" : "rgba(17, 24, 39, 0.25)"}`
          : undefined,
        display: "inline-flex",
      }}
    >
      <Flex alignItems="center" justifyContent="center" width="100%">
        <input
          ref={inputRef}
          aria-label="Frame number"
          inputMode="numeric"
          pattern="[0-9]*"
          value={draftValue}
          onChange={(event) => {
            const next = event.target.value.replace(/\D+/g, "");
            setDraftValue(next);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commitEdit();
              return;
            }

            if (event.key === "Escape") {
              event.preventDefault();
              cancelEdit();
            }
          }}
          onFocus={() => setIsFocused(true)}
          onBlur={cancelEdit}
          style={{
            background: "transparent",
            border: 0,
            color: isDark ? "#ffffff" : "#111827",
            fontSize: 14,
            fontWeight: 600,
            outline: "none",
            padding: 0,
            textAlign: "center",
            width: "100%",
          }}
        />
      </Flex>
    </View>
  );
}

function RectangleIcon() {
  return <ImageMapRectangle aria-hidden="true" size="M" />;
}

function PolygonIcon() {
  return <ImageMapPolygon aria-hidden="true" size="M" />;
}

function PointIcon() {
  return <Target aria-hidden="true" size="M" />;
}

function CanvasModeToggle(props: {
  label: string;
  isSelected: boolean;
  isDisabled?: boolean;
  onPress: () => void;
  children: React.ReactNode;
}) {
  return (
    <TooltipTrigger delay={350} placement="left">
      <ToggleButton
        isSelected={props.isSelected}
        isDisabled={props.isDisabled}
        onPress={props.onPress}
        aria-label={props.label}
        width={40}
        minWidth={40}
        height={40}
        UNSAFE_style={{ padding: 0 }}
      >
        {props.children}
      </ToggleButton>
      <Tooltip>{props.label}</Tooltip>
    </TooltipTrigger>
  );
}

function CanvasActionButton(props: {
  label: string;
  isDisabled?: boolean;
  onPress: () => void;
  children: React.ReactNode;
}) {
  return (
    <TooltipTrigger delay={350} placement="left">
      <ActionButton
        onPress={props.onPress}
        aria-label={props.label}
        isDisabled={props.isDisabled}
        width={40}
        minWidth={40}
        height={40}
        UNSAFE_style={{ padding: 0 }}
      >
        {props.children}
      </ActionButton>
      <Tooltip>{props.label}</Tooltip>
    </TooltipTrigger>
  );
}

export function CanvasModeToolbar(props: {
  panMode: boolean;
  drawingTool: "rectangle" | "polygon" | "point";
  hideAnnotations: boolean;
  onTogglePanMode: () => void;
  onSelectRectangle: () => void;
  onSelectPolygon: () => void;
  onSelectPoint: () => void;
  onResetView: () => void;
}) {
  const { colorScheme } = useProvider();
  const isDark = colorScheme === "dark";
  const isEditMode = !props.panMode;
  const modeLabel = isEditMode
    ? "Edit mode. Click to return to view mode."
    : "View mode. Click to enter edit mode, or hold Shift to edit temporarily.";

  return (
    <View
      position="absolute"
      top="size-100"
      left="calc(100% + 12px)"
      zIndex={20}
    >
      <View
        borderWidth="thin"
        borderColor={isDark ? "gray-700" : "gray-400"}
        borderRadius="large"
        backgroundColor={isDark ? "gray-900" : "gray-100"}
        padding="size-100"
        UNSAFE_style={{
          backgroundImage: isDark
            ? "linear-gradient(180deg, rgba(39, 39, 42, 0.85) 0%, rgba(9, 9, 11, 0.9) 100%)"
            : undefined,
          backdropFilter: "blur(6px)",
          boxShadow: "0 1px 2px rgba(0, 0, 0, 0.08)",
        }}
      >
        <Flex direction="column" alignItems="center" gap="size-100">
          <CanvasModeToggle
            label={modeLabel}
            isSelected={isEditMode}
            isDisabled={props.hideAnnotations}
            onPress={props.onTogglePanMode}
          >
            <Draw aria-hidden="true" size="S" />
          </CanvasModeToggle>
          <CanvasActionButton label="Reset view" onPress={props.onResetView}>
            <FullScreenExit aria-hidden="true" size="S" />
          </CanvasActionButton>
          <Divider size="S" width="100%" />
          <CanvasModeToggle
            label="Rectangle"
            isSelected={props.drawingTool === "rectangle"}
            onPress={props.onSelectRectangle}
          >
            <RectangleIcon />
          </CanvasModeToggle>
          <CanvasModeToggle
            label="Polygon"
            isSelected={props.drawingTool === "polygon"}
            onPress={props.onSelectPolygon}
          >
            <PolygonIcon />
          </CanvasModeToggle>
          <CanvasModeToggle
            label="Point"
            isSelected={props.drawingTool === "point"}
            onPress={props.onSelectPoint}
          >
            <PointIcon />
          </CanvasModeToggle>
        </Flex>
      </View>
    </View>
  );
}

/** Minimal shape used to render/select a tracked instance in the sidebar list. */
export type Profile = {
  key: string;
  class_id: number;
  class_name: string;
  track_id: string;
  first_frame?: number | null;
};

/**
 * Instance list + optional "profile creator" UI.
 * - Selecting an instance calls `onSelect`.
 * - Jump control calls `onJumpToFirstFrame`.
 * - Right-clicking an instance calls `onRequestBulkDelete`.
 * - The creator UI is gated by `showCreator` and `classItems`.
 */
export function InstancePanel({
  profiles,
  selectedKey,
  onSelect,
  onJumpToFirstFrame,
  onCreateProfile,
  onRequestBulkDelete,
  onRequestDeleteAllInstances,
  profileCounts,
  showCreator = true,
  // Only used when showCreator=true
  classItems,
}: {
  profiles: Profile[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  onJumpToFirstFrame?: (profile: Profile) => void;
  onCreateProfile: (className: string, trackId: string) => void;
  onRequestBulkDelete: (profile: Profile) => void;
  onRequestDeleteAllInstances: () => void;
  profileCounts?: Record<string, number>;
  showCreator?: boolean;
  classItems?: { name: string }[];
}) {
  const [open, setOpen] = useState(false);

  // Creator UI state (only meaningful when showCreator=true)
  const defaultClass = classItems?.[0]?.name ?? "";
  const [className, setClassName] = useState<string>(defaultClass);

  const makeAutoTrackId = () =>
    `auto-${Math.random().toString(36).slice(2, 7)}`;
  const [trackId, setTrackId] = useState<string>(() => makeAutoTrackId());

  const creatorEnabled = Boolean(classItems && classItems.length > 0);

  return (
    <div className="w-48 shrink-0 mx-auto">
      {showCreator && (
        <div className="mb-3">
          <Button
            onPress={() => {
              setOpen((prev) => {
                const next = !prev;
                if (next) setTrackId(makeAutoTrackId());
                return next;
              });
            }}
            isDisabled={!creatorEnabled}
            width="100%"
            variant="secondary"
          >
            Add Profile
          </Button>

          {!creatorEnabled && (
            <div className="mt-1 text-[11px] text-gray-500 dark:text-gray-400">
              Creator disabled (no class items provided).
            </div>
          )}
        </div>
      )}

      <button
        onClick={onRequestDeleteAllInstances}
        disabled={profiles.length === 0}
        className={`mb-2 w-full rounded-lg px-2.5 py-1.5 text-left border shadow-sm ${
          profiles.length
            ? "border-red-300 bg-white text-red-700 hover:border-red-400 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-400/30 dark:border-red-500/70 dark:bg-gray-950 dark:text-red-300 dark:hover:border-red-400 dark:hover:bg-red-950/40"
            : "cursor-not-allowed border-gray-200 bg-gray-100 text-gray-400 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-500"
        }`}
        title="Delete all instances and their annotations across all frames"
      >
        <span className="text-sm">Delete All Instances</span>
      </button>

      {showCreator && open && (
        <div className="mt-2 space-y-2 rounded-lg border border-gray-200 bg-white p-2 text-gray-900 shadow-sm dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100">
          <div>
            <label className="text-xs text-gray-600 dark:text-gray-300">
              Class Label
            </label>
            <select
              value={className}
              onChange={(e) => setClassName(e.target.value)}
              className="mt-1 w-full rounded border border-gray-300 bg-white px-2 py-1.5 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-orange-400/40 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
            >
              {(classItems ?? []).map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-600 dark:text-gray-300">
              Track ID
            </label>
            <input
              type="text"
              value={trackId}
              readOnly
              className="mt-1 w-full cursor-not-allowed select-all rounded border border-gray-300 bg-gray-100 px-2 py-1 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200"
              title="Auto-generated; not editable"
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => {
                const name = (className || "").trim();
                const trk = (trackId || "").trim();
                if (!name || !trk) return;
                onCreateProfile(name, trk);
                setOpen(false);
              }}
              className="flex-1 rounded-md bg-orange-500 hover:bg-orange-600 text-white px-2.5 py-1.5 text-xs"
              disabled={!creatorEnabled}
            >
              Create
            </button>
            <button
              onClick={() => setOpen(false)}
              className="rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-xs text-gray-700 hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-950 dark:text-gray-100 dark:hover:bg-gray-900"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="mt-2 max-h-[45vh] overflow-y-auto rounded-lg border border-gray-200 bg-white/90 shadow-sm dark:border-gray-700 dark:bg-gray-950/70">
        {profiles.length === 0 && (
          <div className="p-3 text-sm text-gray-600 dark:text-gray-300">
            {showCreator
              ? "No profiles yet. Click “Add Profile”."
              : "No instances yet. Pick a class above, then draw to create one."}
          </div>
        )}

        {profiles.map((p) => {
          const count = profileCounts?.[p.key] ?? 0;
          const canJumpToFirstFrame = Boolean(
            onJumpToFirstFrame && p.first_frame != null,
          );

          return (
            <button
              key={p.key}
              type="button"
              onClick={() => onSelect(p.key)}
              onContextMenu={(e) => {
                e.preventDefault();
                onRequestBulkDelete(p);
              }}
              className={`w-full text-left px-2 py-1.5 border-b last:border-b-0 transition leading-snug ${
                selectedKey === p.key
                  ? "border-orange-200 bg-orange-50 ring-1 ring-orange-400/40 dark:border-gray-600 dark:bg-gray-900 dark:ring-orange-400/60"
                  : "border-gray-200 bg-transparent hover:bg-gray-50 dark:border-gray-800 dark:hover:bg-gray-900"
              } text-gray-900 dark:text-gray-100`}
              title={`Select: ${p.class_name} (${p.track_id}). Use the rewind button to jump to first frame. Right-click to bulk delete.`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-xs font-semibold text-gray-900 dark:text-gray-100">
                    #{p.track_id}
                  </div>
                  <div className="mt-0 text-[11px] text-gray-600 dark:text-gray-300">
                    Class: {p.class_name}
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span
                    className="inline-block rounded-full border border-gray-200 bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-700 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200"
                    title="Total annotations for this instance across all frames"
                  >
                    {count}
                  </span>

                  <div className="w-14 flex flex-col items-stretch gap-1 shrink-0">
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onRequestBulkDelete(p);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          e.stopPropagation();
                          onRequestBulkDelete(p);
                        }
                      }}
                      className="w-full cursor-pointer select-none rounded-md border border-red-300 px-2 py-1 text-center text-[10px] text-red-700 hover:bg-red-50 dark:border-red-400/60 dark:text-red-200 dark:hover:bg-red-500/15"
                      title="Delete this instance across all frames"
                      aria-label={`Delete ${p.class_name} ${p.track_id}`}
                    >
                      Delete
                    </span>

                    <span
                      role="button"
                      tabIndex={canJumpToFirstFrame ? 0 : -1}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (!canJumpToFirstFrame) return;
                        onJumpToFirstFrame?.(p);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          e.stopPropagation();
                          if (!canJumpToFirstFrame) return;
                          onJumpToFirstFrame?.(p);
                        }
                      }}
                      className={`w-full rounded-md px-1.5 py-0.5 text-[10px] border inline-flex items-center justify-center select-none ${
                        canJumpToFirstFrame
                          ? "cursor-pointer border-gray-300 text-gray-700 hover:bg-gray-100 dark:border-gray-400/60 dark:text-gray-100 dark:hover:bg-gray-500/15"
                          : "cursor-not-allowed border-gray-200 text-gray-300 dark:border-gray-800 dark:text-white/30"
                      }`}
                      title={
                        canJumpToFirstFrame
                          ? "Jump to the first frame where this instance appears"
                          : "No known first frame for this instance"
                      }
                      aria-label={
                        canJumpToFirstFrame
                          ? `Jump to first frame for ${p.class_name} ${p.track_id}`
                          : `No first frame available for ${p.class_name} ${p.track_id}`
                      }
                    >
                      <StepBackward aria-hidden="true" size="XS" />
                    </span>
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Simple confirmation modal used by destructive actions (bulk deletes, etc.).
 * Kept dependency-free to avoid quirks across UI libraries.
 */
export function ConfirmModal({
  open,
  title,
  message,
  details,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  details?: React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/40 px-4">
      <div className="w-full max-w-md rounded-xl border border-gray-200 bg-white text-gray-900 shadow-xl dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100">
        <div className="border-b border-gray-200 px-4 py-3 dark:border-gray-700">
          <h2 className="text-base font-semibold">{title}</h2>
        </div>

        <div className="px-4 py-3 space-y-2">
          <p className="text-sm text-gray-700 dark:text-gray-200">{message}</p>
          {details && (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {details}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-gray-200 px-4 py-3 dark:border-gray-700">
          <button
            onClick={onCancel}
            className="rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-100 dark:hover:bg-gray-800"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className="px-3 py-1.5 rounded-md bg-red-600 hover:bg-red-700 text-white text-sm"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
