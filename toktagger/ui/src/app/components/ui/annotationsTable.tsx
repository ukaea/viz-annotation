"use client";

import { TimeSeriesAnnotationType, TimeSeriesCategory } from "@/types";
import { ComponentType, useMemo } from "react";
import {
  TableView,
  TableHeader,
  Column,
  TableBody,
  Row,
  Cell,
  Flex,
} from "@adobe/react-spectrum";
import { useTimeSeriesState } from "@/app/contexts/TimeSeriesContext";

interface MarkerProps {
  color: string;
}

const MARKER_VIEWBOX = "0 0 24 24";

// Shape mirrors how each annotation type actually renders on the plot, so the
// marker doubles as a legend for the tool - stroked/filled with the category color
const TimePointMarker = ({ color }: MarkerProps) => (
  <svg width="10" height="20" viewBox={MARKER_VIEWBOX}>
    <line x1="12" y1="1" x2="12" y2="40" stroke={color} strokeWidth="4" />
  </svg>
);

const TimeRegionMarker = ({ color }: MarkerProps) => (
  <svg width="20" height="20" viewBox={MARKER_VIEWBOX}>
    <rect x="6" y="6" width="12" height="20" rx="2" fill={color} />
  </svg>
);

const BoundingBoxMarker = ({ color }: MarkerProps) => (
  <svg width="20" height="20" viewBox={MARKER_VIEWBOX}>
    <rect
      x="3"
      y="5"
      width="18"
      height="14"
      rx="1"
      fill="none"
      stroke={color}
      strokeWidth="2.5"
    />
  </svg>
);

const PolygonMarker = ({ color }: MarkerProps) => (
  <svg width="20" height="20" viewBox={MARKER_VIEWBOX}>
    <polygon
      points="12,2 22,9.5 18,21 6,21 2,9.5"
      fill="none"
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
  </svg>
);

// One marker per annotation type - set when the annotation's category/type is resolved below
const MARKER_ICONS: Record<TimeSeriesAnnotationType, ComponentType<MarkerProps>> = {
  [TimeSeriesAnnotationType.TIME_POINT]: TimePointMarker,
  [TimeSeriesAnnotationType.TIME_REGION]: TimeRegionMarker,
  [TimeSeriesAnnotationType.BOUNDING_BOX]: BoundingBoxMarker,
  [TimeSeriesAnnotationType.POLYGON]: PolygonMarker,
};

interface TableEntry {
  id: string;
  category: TimeSeriesCategory;
  data: string;
  marker: ComponentType<MarkerProps>;
}

export const AnnotationsTable = () => {
  const { annotations, categories } = useTimeSeriesState();

  const entries = useMemo<TableEntry[]>(() => {
    const entriesBuffer: TableEntry[] = [];
    annotations.forEach((annotation) => {
      const categoryId = `${annotation.type}_${annotation.label}`;
      const category = categories.get(categoryId);
      if (!category) {
        console.error(
          `Could not locate ${categoryId} when assigning table entry`,
        );
        return;
      }

      let data: string;
      switch (annotation.type) {
        case TimeSeriesAnnotationType.TIME_POINT:
          data = `${annotation.points[0].x.toFixed(4)}`;
          break;
        case TimeSeriesAnnotationType.TIME_REGION: {
          const timeRegionPoints: string[] = [];
          annotation.points.forEach((point) => {
            timeRegionPoints.push(`${point.x.toFixed(4)}`);
          });
          data = `${timeRegionPoints[0]} - ${timeRegionPoints[1]}`;
          break;
        }
        case TimeSeriesAnnotationType.BOUNDING_BOX: {
          const boundingBoxPoints: string[] = [];
          annotation.points.forEach((point) => {
            boundingBoxPoints.push(`(${point.x.toFixed(2)}, ${point.y.toFixed(2)})`)
          })
          data = `${boundingBoxPoints[0]} ${boundingBoxPoints[1]}`;
          break;
        }
        case TimeSeriesAnnotationType.POLYGON:
          data = `(${annotation.points[0].x.toFixed(2)}, ${annotation.points[0].y.toFixed(2)}) [${annotation.points.length}]`;
          break;
        case TimeSeriesAnnotationType.BOUNDING_BOX:
          marker = {
            width: "size-75",
            height: "size-250",
          };
          data = `${annotation.points[0].x.toFixed(4)}`;
          break;
        default:
          console.warn(
            `Could not parse data for ${annotation.type} when adding to table`,
          );
          data = "";
      }

      entriesBuffer.push({
        id: annotation.id,
        category,
        data,
        marker: MARKER_ICONS[annotation.type],
      });
    });

    return entriesBuffer;
  }, [annotations, categories]);

  return (
    <div className="relative w-[70%] overflow-x-auto shadow-md sm:rounded-lg ml-auto mr-auto p-4">
      {/* <ToolingControls /> */}
      <Flex justifyContent="center" marginBottom="size-200">
        <h1 className="text-xl font-bold">Annotations</h1>
      </Flex>
      <TableView aria-label="Annotations table" width="100%" height="200px">
        <TableHeader>
          <Column key="marker" width="2%">
            <></>
          </Column>
          <Column key="category" width="28%">
            Category
          </Column>
          <Column key="type" width="20%">
            Type
          </Column>
          <Column key="data" width="50%">
            Data
          </Column>
        </TableHeader>
        <TableBody items={entries}>
          {(item: TableEntry) => (
            <Row key={item.id}>
              <Cell>
                <Flex justifyContent="center">
                  <item.marker color={item.category.color} />
                </Flex>
              </Cell>
              <Cell>
                <span>{item.category.label}</span>
              </Cell>
              <Cell>{item.category.type}</Cell>
              <Cell>{item.data}</Cell>
            </Row>
          )}
        </TableBody>
      </TableView>
    </div>
  );
};
