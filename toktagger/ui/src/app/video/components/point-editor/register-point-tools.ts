import {
  ShapeType,
  type AnnotoriousOpenSeadragonAnnotator,
} from "@annotorious/react";
import type { ComponentConstructorOptions, SvelteComponent } from "svelte";
import { asClassComponent } from "svelte/legacy";

import PointEditor from "./PointEditor.svelte";

type PointEditorComponentOptions = ComponentConstructorOptions<
  Record<string, unknown>
>;

type PointEditorComponentInstance = SvelteComponent & {
  $$set: (props: Record<string, unknown>) => void;
};

const LegacyPointEditor = asClassComponent(PointEditor) as new (
  options: PointEditorComponentOptions,
) => PointEditorComponentInstance;

class PointEditorComponent extends LegacyPointEditor {
  constructor(options: PointEditorComponentOptions) {
    super(options);
    this.$$set = (props: Record<string, unknown>) => this.$set(props);
  }
}

export function registerPointEditor(
  api: Pick<AnnotoriousOpenSeadragonAnnotator, "registerShapeEditor">,
) {
  api.registerShapeEditor(
    ShapeType.ELLIPSE,
    PointEditorComponent as typeof SvelteComponent,
  );
}
