"use client";

import { useEffect, useMemo, useState } from "react";
import {
  DialogContainer,
  AlertDialog,
  Switch,
  Divider,
  Flex,
} from "@adobe/react-spectrum";

import { useVideoSession } from "@/app/video/components/video-session";
import { canonicalizeTrackId } from "@/app/video/components/video-utils";
import { useSample } from "@/app/contexts/SampleContext";
import { useVideoUiState } from "@/app/contexts/VideoContext";
import {
  ClassPanel as VideoClassPanel,
  InstancePanel as VideoInstancePanel,
  ConfirmModal,
} from "@/app/video/components/ui_elements";

/**
 * Sidebar instance rows are keyed by (class, track_id). We keep a stable string key
 * for selection, sorting, and count lookups.
 */
function instanceKey(args: { class_name: string; track_id: string }) {
  const cls = (args.class_name || "").toLowerCase();
  const tid = canonicalizeTrackId(args.track_id || "");
  return `${cls}:${tid}`;
}

export function VideoToolbox() {
  const session = useVideoSession();
  const { annotationLabels, dataParams, setDataParams } = useSample();
  const { videoLastClassName, setVideoLastClassName } = useVideoUiState();
  const labels = annotationLabels;

  const [confirmClearAllOpen, setConfirmClearAllOpen] = useState(false);

  const [pendingDeleteInstance, setPendingDeleteInstance] = useState<{
    className: string;
    trackId: string;
  } | null>(null);

  // Restore the last selected class, or default to the first configured label.
  useEffect(() => {
    if (session.selection.className) return;

    const firstClassName = labels[0]?.name ?? null;
    const lastClassName =
      videoLastClassName &&
      labels.some((label) => label.name === videoLastClassName)
        ? videoLastClassName
        : null;
    const nextClassName = lastClassName ?? firstClassName;

    if (!nextClassName) return;

    session.setSelection({
      className: nextClassName,
      trackId: null,
      source: null,
    });

    if (nextClassName !== videoLastClassName) {
      setVideoLastClassName(nextClassName);
    }
  }, [labels, session, setVideoLastClassName, videoLastClassName]);

  const classItems = useMemo(() => {
    return labels.map((c) => ({ name: c.name }));
  }, [labels]);

  const classIdByName = useMemo(() => {
    const map = new Map<string, number>();
    for (const c of labels) map.set(c.name, c.id);
    return map;
  }, [labels]);

  const profiles = useMemo(() => {
    return session.instances.map((inst) => ({
      key: instanceKey({ class_name: inst.className, track_id: inst.trackId }),
      class_id: classIdByName.get(inst.className) ?? inst.classId ?? -1,
      class_name: inst.className,
      track_id: canonicalizeTrackId(inst.trackId),
      first_frame: inst.frames[0] ?? null,
    }));
  }, [session.instances, classIdByName]);

  const profileCounts = useMemo(() => {
    const out: Record<string, number> = {};
    for (const inst of session.instances) {
      const k = instanceKey({
        class_name: inst.className,
        track_id: inst.trackId,
      });
      out[k] = inst.count;
    }
    return out;
  }, [session.instances]);

  const selectedKey = useMemo(() => {
    if (!session.selection.className || !session.selection.trackId) return null;
    return instanceKey({
      class_name: session.selection.className,
      track_id: session.selection.trackId,
    });
  }, [session.selection.className, session.selection.trackId]);

  const onSelectClassName = (name: string | null) => {
    const cls = (name ?? "").trim();
    if (!cls) return;

    session.setSelection({ className: cls, trackId: null, source: "explicit" });
    setVideoLastClassName(cls);
  };

  const onSelectInstance = (key: string) => {
    if (!key) {
      session.setSelection({
        className: session.selection.className ?? null,
        trackId: null,
        source: "explicit",
      });
      return;
    }

    const hit = profiles.find((p) => p.key === key);
    if (!hit) return;

    if (selectedKey === key) {
      session.setSelection({
        className: hit.class_name,
        trackId: null,
        source: "explicit",
      });
      session.closePopup();
      setVideoLastClassName(hit.class_name);
      return;
    }

    session.setSelection({
      className: hit.class_name,
      trackId: hit.track_id,
      source: "explicit",
    });
    session.requestFocusInstance(hit.class_name, hit.track_id, {
      onlyIfOnCurrentFrame: true,
    });

    setVideoLastClassName(hit.class_name);
  };

  const onJumpToFirstFrame = (profile: {
    class_name?: string;
    track_id?: string;
    first_frame?: number | null;
  }) => {
    const cls = (profile.class_name || "").trim();
    const tid = canonicalizeTrackId(profile.track_id || "");
    if (!cls || !tid) return;

    session.setSelection({
      className: cls,
      trackId: tid,
      source: "explicit",
    });
    setVideoLastClassName(cls);

    const firstFrame = profile.first_frame;
    if (typeof firstFrame === "number" && Number.isFinite(firstFrame)) {
      setDataParams({
        ...dataParams,
        name: "image",
        frame: Math.max(0, Math.trunc(firstFrame)),
      });
    }

    session.requestFocusInstance(cls, tid, {
      targetFrame:
        typeof firstFrame === "number" && Number.isFinite(firstFrame)
          ? Math.max(0, Math.trunc(firstFrame))
          : undefined,
    });
  };

  const onRequestBulkDelete = (profile: {
    class_name?: string;
    track_id?: string;
  }) => {
    const cls = (profile.class_name || "").trim();
    const tid = canonicalizeTrackId(profile.track_id || "");
    if (!cls || !tid) return;

    setPendingDeleteInstance({ className: cls, trackId: tid });
  };

  const cancelDeleteInstance = () => setPendingDeleteInstance(null);

  const confirmDeleteInstance = () => {
    const pending = pendingDeleteInstance;
    if (!pending) return;

    session.deleteInstanceAcrossFrames(pending.className, pending.trackId);
    setPendingDeleteInstance(null);
  };

  const onRequestDeleteAllInstances = () => setConfirmClearAllOpen(true);

  const confirmClearAll = () => {
    setConfirmClearAllOpen(false);
    session.clearAllFrames();
  };

  const cancelClearAll = () => setConfirmClearAllOpen(false);

  return (
    <>
      <div className="w-full">
        <Divider size="S" marginX="size-200" />
        <div className="px-4 py-4">
          <div className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-200">
            Frame Tools
          </div>
          <Flex
            alignItems="center"
            justifyContent="center"
            direction="column"
            gap="size-100"
          >
            <div className="w-[170px] flex justify-start">
              <Switch
                isSelected={session.propagate}
                isDisabled={!session.editMode}
                onChange={session.setPropagate}
              >
                Propagation
              </Switch>
            </div>
            <div className="w-[170px] flex justify-start">
              <Switch
                isSelected={session.hideAnnotations}
                onChange={session.setHideAnnotations}
              >
                <span className="whitespace-nowrap">Hide annotations</span>
              </Switch>
            </div>
          </Flex>
        </div>
        <Divider size="S" marginX="size-200" />

        <div className="px-4 py-4">
          <VideoClassPanel
            items={classItems}
            selectedClassName={session.selection.className}
            setSelectedClassName={onSelectClassName}
          />
        </div>
        <Divider size="S" marginX="size-200" />

        <div className="px-4 py-4">
          <div className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-200">
            Instances
          </div>
          <VideoInstancePanel
            profiles={profiles}
            selectedKey={selectedKey}
            onSelect={onSelectInstance}
            onJumpToFirstFrame={onJumpToFirstFrame}
            onCreateProfile={() => {
              // Instances are derived from annotations; creation happens via drawing.
            }}
            onRequestBulkDelete={onRequestBulkDelete}
            onRequestDeleteAllInstances={onRequestDeleteAllInstances}
            profileCounts={profileCounts}
            showCreator={false}
            classItems={classItems}
          />
        </div>
      </div>

      <DialogContainer onDismiss={cancelClearAll}>
        {confirmClearAllOpen && (
          <AlertDialog
            title="Clear all frames?"
            variant="destructive"
            primaryActionLabel="Clear all"
            cancelLabel="Cancel"
            onPrimaryAction={confirmClearAll}
            onCancel={cancelClearAll}
          >
            This will remove all annotations across all frames in the current
            session. You can’t undo this.
          </AlertDialog>
        )}
      </DialogContainer>

      <ConfirmModal
        open={Boolean(pendingDeleteInstance)}
        title="Delete instance?"
        message="This deletes all annotations for this instance across all frames. You can’t undo this."
        details={
          pendingDeleteInstance ? (
            <div>
              Target:{" "}
              <b>
                {pendingDeleteInstance.className} /{" "}
                {pendingDeleteInstance.trackId}
              </b>
            </div>
          ) : null
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={confirmDeleteInstance}
        onCancel={cancelDeleteInstance}
      />
    </>
  );
}
