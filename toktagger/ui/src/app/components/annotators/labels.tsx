import React, { useState, useEffect, useCallback } from "react";
import { ListView, Item } from "@adobe/react-spectrum";
import { Annotation, ClassLabel } from "@/types";
import { Selection } from "@react-types/shared";
import { useSample } from "@/app/contexts/SampleContext";
import { useAuth } from "@/app/contexts/AuthContext";

export type ShotLabelsType = {
  labels: string[];
  // Supplied by the toolbar, which already resolves the project role, so this does
  // not cost another membership request.
  canAnnotate?: boolean;
};

export function ShotLabels({
  labels = [],
  canAnnotate = true,
}: ShotLabelsType) {
  const { project, sample, annotations, setAnnotations } = useSample();
  const { user } = useAuth();
  const items = labels.map((label, index) => ({ id: index, name: label }));
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    const defaultAnnotations = annotations.filter(
      (annotation: Annotation) => annotation.type === "class_label",
    );
    const defaultSelectedKeys = defaultAnnotations
      .map((annotation: Annotation) => {
        const index = labels.indexOf(annotation.label);
        return index !== -1 ? index.toString() : null;
      })
      .filter((key) => key !== null) as string[];

    setSelectedKeys(new Set(defaultSelectedKeys));
  }, [annotations, setSelectedKeys, labels]);

  const onSelectionChange = useCallback(
    (keys: Selection) => {
      // Gate the handler, not just the ListView: the number-key shortcut below
      // routes through here too, so a viewer could otherwise relabel with the
      // keyboard even while the list is disabled.
      if (!canAnnotate) return;
      let newKeys = new Set<string>();
      if (keys === "all") {
        items.forEach((item) => newKeys.add(item.id.toString()));
      } else {
        newKeys = new Set(Array.from(keys).map((key) => key.toString()));
      }

      setAnnotations((prevAnnotations: Annotation[]) => {
        let newAnnotations = prevAnnotations || [];
        newAnnotations = newAnnotations.filter(
          (annotation) => annotation.type !== "class_label",
        );
        newKeys.forEach((key: string) => {
          const item = items.find((item) => item.id.toString() === key) || null;

          if (item === null) {
            console.warn(`Label with key ${key} not found in items.`);
            return;
          }

          newAnnotations.push({
            project_id: project?._id,
            sample_id: sample?._id,
            shot_id: sample?.shot_id,
            type: "class_label",
            label: item.name,
            // Authored by whoever selected the label; the server stamps the same
            // username on save.
            created_by: user?.username ?? "manual",
          } as ClassLabel);
        });
        return newAnnotations;
      });
    },
    [project, sample, items, setAnnotations, canAnnotate, user],
  );

  useEffect(() => {
    const handleKeyDown = (e: { key: string }) => {
      const key = e.key.toLowerCase();
      const matchedItem = items.find((item) => item.id.toString() === key);
      if (matchedItem) {
        if (selectedKeys.has(matchedItem.id.toString())) {
          selectedKeys.delete(matchedItem.id.toString());
        } else {
          selectedKeys.add(matchedItem.id.toString());
        }
        onSelectionChange(selectedKeys);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [items, selectedKeys, onSelectionChange]);

  if (items.length === 0) {
    return (
      <div>
        No labels available. Please define labels in the project settings.
      </div>
    );
  }

  return (
    <>
      <ListView
        items={items}
        selectedKeys={selectedKeys}
        onSelectionChange={onSelectionChange}
        selectionMode={canAnnotate ? "multiple" : "none"}
        aria-label="Labels"
        maxWidth="size-6000"
      >
        {(item) => (
          <Item key={item.id} textValue={item.name}>
            {`${item.id} | ${item.name}`}
          </Item>
        )}
      </ListView>
    </>
  );
}
