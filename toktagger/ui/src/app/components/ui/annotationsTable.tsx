"use client";

import { TimeSeriesAnnotationType, TimeSeriesCategory } from "@/types";
import { useMemo } from "react";
import {
  TableView,
  TableHeader,
  Column,
  TableBody,
  Row,
  Cell,
  Flex,
  View,
  DimensionValue,
} from "@adobe/react-spectrum";
import { useTimeSeriesState } from "@/app/contexts/TimeSeriesContext";

interface TableEntry {
  id: string;
  category: TimeSeriesCategory;
  data: string;
  created_by: string;
  marker: {
    width: DimensionValue;
    height: DimensionValue;
  };
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
      let marker: TableEntry["marker"] = {
        width: "size-250",
        height: "size-250",
      };
      switch (annotation.type) {
        case TimeSeriesAnnotationType.TIME_POINT:
          marker = {
            width: "size-75",
            height: "size-250",
          };
          data = `${annotation.points[0].x.toFixed(4)}`;
          break;
        case TimeSeriesAnnotationType.TIME_REGION:
          marker = {
            width: "size-250",
            height: "size-250",
          };
          const points: string[] = [];
          annotation.points.forEach((point) => {
            points.push(`${point.x.toFixed(4)}`);
          });
          data = `${points[0]} - ${points[1]}`;
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
        created_by: annotation.created_by,
        marker,
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
          <Column key="category" width="23%">
            Category
          </Column>
          <Column key="type" width="17%">
            Type
          </Column>
          <Column key="data" width="38%">
            Data
          </Column>
          <Column key="created_by" width="20%">
            Created by
          </Column>
        </TableHeader>
        <TableBody items={entries}>
          {(item: TableEntry) => (
            <Row key={item.id}>
              <Cell>
                <Flex justifyContent="center">
                  <View
                    width={item.marker.width}
                    height={item.marker.height}
                    borderRadius="small"
                    UNSAFE_style={{ backgroundColor: item.category.color }}
                  />
                </Flex>
              </Cell>
              <Cell>
                <span>{item.category.label}</span>
              </Cell>
              <Cell>{item.category.type}</Cell>
              <Cell>{item.data}</Cell>
              <Cell>{item.created_by}</Cell>
            </Row>
          )}
        </TableBody>
      </TableView>
    </div>
  );
};
