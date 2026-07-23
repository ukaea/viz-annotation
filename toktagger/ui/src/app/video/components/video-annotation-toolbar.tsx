"use client";

import { useEffect, useState } from "react";
import {
  ActionButton,
  Button,
  Content,
  ContextualHelp,
  Divider,
  Flex,
  Heading,
  Text,
  ToggleButton,
  Tooltip,
  TooltipTrigger,
  View,
} from "@adobe/react-spectrum";
import FullScreenExit from "@spectrum-icons/workflow/FullScreenExit";
import ImageMapPolygon from "@spectrum-icons/workflow/ImageMapPolygon";
import ImageMapRectangle from "@spectrum-icons/workflow/ImageMapRectangle";
import Target from "@spectrum-icons/workflow/Target";
import {
  useAnnotator,
  type AnnotoriousOpenSeadragonAnnotator,
} from "@annotorious/react";

import { useVideoSession } from "./video-session";
import type { DrawingTool } from "./types";

export function VideoAnnotationToolbar() {
  const {
    drawingTool,
    setDrawingTool,
    editMode,
    setEditMode,
    hideAnnotations,
  } = useVideoSession();
  const api = useAnnotator<AnnotoriousOpenSeadragonAnnotator>();
  const [hasEnteredEditMode, setHasEnteredEditMode] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    if (!editMode || hasEnteredEditMode) return;
    setHasEnteredEditMode(true);
    setHelpOpen(true);
  }, [editMode, hasEnteredEditMode]);

  const toggleTool = (tool: DrawingTool) => {
    setDrawingTool(drawingTool === tool ? null : tool);
  };

  const resetView = () => {
    const viewport = api?.viewer?.viewport;
    if (!viewport) return;
    viewport.goHome(true);
    viewport.applyConstraints();
  };

  const toolsDisabled = !editMode || hideAnnotations;
  const modeVariant: "accent" | "primary" = editMode ? "accent" : "primary";

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
            isDisabled={hideAnnotations}
            onPress={() => setEditMode(!editMode)}
          >
            {editMode ? "Edit Mode" : "View Mode"}
          </Button>
          <Tooltip>
            {editMode
              ? "Return to View Mode (shortcut: e)"
              : "Enter Edit Mode (shortcut: e)"}
          </Tooltip>
        </TooltipTrigger>

        <Divider size="S" marginX="size-200" />
        <h1 className="text-xl font-bold">Tools</h1>

        <Flex direction="column" alignItems="center" gap="size-100">
          <TooltipTrigger placement="left">
            <ToggleButton
              width="size-1800"
              isDisabled={toolsDisabled}
              isSelected={drawingTool === "rectangle"}
              onPress={() => toggleTool("rectangle")}
            >
              <Flex width="100%" alignItems="center" gap="size-100">
                <View width="size-250" flexShrink={0} paddingStart="size-50">
                  <ImageMapRectangle aria-hidden="true" size="S" />
                </View>
                <Text>Bounding Box</Text>
              </Flex>
            </ToggleButton>
            <Tooltip>Draw a bounding box with Ctrl + drag</Tooltip>
          </TooltipTrigger>

          <TooltipTrigger placement="left">
            <ToggleButton
              width="size-1600"
              isDisabled={toolsDisabled}
              isSelected={drawingTool === "polygon"}
              onPress={() => toggleTool("polygon")}
            >
              <Flex width="100%" alignItems="center" gap="size-100">
                <View width="size-250" flexShrink={0} paddingStart="size-50">
                  <ImageMapPolygon aria-hidden="true" size="S" />
                </View>
                <Text>Polygon</Text>
              </Flex>
            </ToggleButton>
            <Tooltip>Draw a polygon with Ctrl + drag</Tooltip>
          </TooltipTrigger>

          <TooltipTrigger placement="left">
            <ToggleButton
              width="size-1600"
              isDisabled={toolsDisabled}
              isSelected={drawingTool === "point"}
              onPress={() => toggleTool("point")}
            >
              <Flex width="100%" alignItems="center" gap="size-100">
                <View width="size-250" flexShrink={0} paddingStart="size-50">
                  <Target aria-hidden="true" size="S" />
                </View>
                <Text>Point</Text>
              </Flex>
            </ToggleButton>
            <Tooltip>Place a point with Ctrl + click</Tooltip>
          </TooltipTrigger>

          <Divider size="S" marginY="size-50" />
          <TooltipTrigger placement="left">
            <ActionButton width="size-1600" onPress={resetView}>
              <Flex width="100%" alignItems="center" gap="size-100">
                <View width="size-250" flexShrink={0} paddingStart="size-50">
                  <FullScreenExit aria-hidden="true" size="S" />
                </View>
                <Text>Reset View</Text>
              </Flex>
            </ActionButton>
            <Tooltip>Reset zoom and pan</Tooltip>
          </TooltipTrigger>
        </Flex>
      </Flex>

      <Flex direction="row" justifyContent="end" marginEnd="size-100">
        <ContextualHelp
          isOpen={helpOpen}
          onOpenChange={setHelpOpen}
          aria-label="annotation-context-help"
        >
          <Heading>Annotation Toolbar</Heading>
          <Content>
            <Text>
              Use the top button or <b>e</b> to switch between view and edit
              modes. Zoom, pan, and select annotations in either mode; shapes
              can only be changed in edit mode.
              <br />
              <br />
              In edit mode, select a tool and hold <b>Ctrl</b> while dragging or
              clicking to create a new annotation. Ctrl drawing takes priority
              over annotations already beneath the pointer.
              <br />
              <br />
              Right-click empty image space to change the selected class label.
            </Text>
          </Content>
        </ContextualHelp>
      </Flex>
    </View>
  );
}
