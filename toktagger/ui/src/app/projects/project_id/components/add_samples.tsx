"use client";
import {
  Form,
  Button,
  DialogTrigger,
  Dialog,
  Divider,
  Heading,
  Content,
  ButtonGroup,
  Text,
  TextField,
  ComboBox,
  Item,
  Flex,
  ToastQueue,
  RadioGroup,
  Radio,
} from "@adobe/react-spectrum";
import {
  Project,
  Sample,
  ShotData,
  FileData,
  TimeSeriesFileData,
  ImageArrayFileData,
} from "@/types";
import AddCircle from "@spectrum-icons/workflow/AddCircle";
import { useState, useEffect } from "react";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import NumericalRange, {
  NumericalRangeType,
} from "@/app/components/ui/numerical_range";

export const AddSamplesEditor = ({
  project,
  onModify,
}: {
  project: Project;
  onModify?: () => void;
}) => {
  const dataLoader = project.data_loader;

  // Data schema state
  const [dataSchema, setDataSchema] = useState<Record<string, unknown> | null>(
    null,
  );
  const [fileTypes, setFileTypes] = useState<{ key: string; value: string }[]>(
    [],
  );

  // Form state
  const [shotInputMethod, setShotInputMethod] = useState<string>("range"); // "range" or "file"
  const [shotRange, setShotRange] = useState<NumericalRangeType>();
  const [shotIds, setShotIds] = useState<number[]>([]);

  // Shot Data fields
  const [signalNames, setSignalNames] = useState<string>("");

  // File Data fields
  const [fileType, setFileType] = useState<string>("parquet");
  const [dirPath, setDirPath] = useState<string>("");
  const [columnNames, setColumnNames] = useState<string>(""); // For time series files

  // Determine if we should use directories (for image data)
  const useDirectories = dataSchema?.title === "ImageFileData";

  // Fetch the data schema type from the load registry
  useEffect(() => {
    async function fetchDataSchema() {
      try {
        const response = await apiFetch(
          `${BACKEND_API_URL}/meta/dataloader/${dataLoader}`,
        );
        if (response.ok) {
          const schema = await response.json();
          // The schema has a "title" field that indicates the type
          // e.g., "ShotData", "FileData", "TimeSeriesFileData"
          setDataSchema(schema);

          if (schema.title === "ShotData") {
            return;
          }

          const fileTypes = schema.properties.type.enum;
          setFileTypes(
            fileTypes.map((ft: string) => ({
              key: ft,
              value: ft.toUpperCase(),
            })),
          );
        } else {
          ToastQueue.negative(`Error fetching data schema for ${dataLoader}.`, {
            timeout: 3000,
          });
        }
      } catch (error) {
        ToastQueue.negative(`Error fetching data schema: ${error}`, {
          timeout: 3000,
        });
      }
    }
    if (dataLoader) {
      fetchDataSchema();
    }
  }, [dataLoader]);

  useEffect(() => {
    if (shotInputMethod === "range") {
      setShotIds(
        Array.from(
          { length: (shotRange?.max ?? 0) - (shotRange?.min ?? 0) + 1 },
          (_, i) => i + (shotRange?.min ?? 0),
        ),
      );
    }
  }, [shotRange, shotInputMethod]);

  // Read and parse CSV file when selected
  const handleFileSelect = (files: FileList | null) => {
    if (files && files.length > 0) {
      const file = files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          // Parse CSV: split by lines, skip header (first line), get first column
          const lines = content.split(/\r?\n/).filter((line) => line.trim());
          if (lines.length <= 1) {
            ToastQueue.negative("CSV file is empty or has no data rows", {
              timeout: 3000,
            });
            return;
          }
          // Skip header (index 0), parse remaining lines
          const ids = lines
            .slice(1)
            .map((line) => {
              // Get first column (split by comma, take first element)
              const firstColumn = line.split(",")[0].trim();
              return parseInt(firstColumn, 10);
            })
            .filter((id) => !isNaN(id));

          if (ids.length === 0) {
            ToastQueue.negative("No valid shot IDs found in first column", {
              timeout: 3000,
            });
          } else {
            setShotIds(ids);
            ToastQueue.positive(`Loaded ${ids.length} shot IDs from CSV`, {
              timeout: 3000,
            });
          }
        } catch (error) {
          ToastQueue.negative(`Error parsing CSV file: ${error}`, {
            timeout: 3000,
          });
        }
      };
      reader.onerror = () => {
        ToastQueue.negative("Error reading file", { timeout: 3000 });
      };
      reader.readAsText(file);
    }
  };

  useEffect(() => {
    async function fetchFileSamples() {
      let apiUrl = `${BACKEND_API_URL}/paths/files?dir_path=${dirPath}&file_type=${fileType}`;
      if (useDirectories) {
        apiUrl = `${BACKEND_API_URL}/paths/directories?dir_path=${dirPath}&file_type=${fileType}`;
      }
      const response = await apiFetch(apiUrl);

      if (response.ok) {
        const result = await response.json();
        const fileNames: string[] = result;

        // Extract shot IDs from file names using regex (assuming shot ID is a number in the file name)
        const extractedShotIds = fileNames
          .map((fileName) => {
            const regex = useDirectories
              ? /^\d+$/
              : new RegExp(`^\\d+\\.${fileType}$`);

            const match = fileName.split("/").pop()?.match(regex);
            return match ? parseInt(match[0], 10) : null;
          })
          .filter((id): id is number => id !== null);

        setShotIds(extractedShotIds);
      } else {
        ToastQueue.negative(`Error fetching files for ${dirPath}.`, {
          timeout: 3000,
        });
      }
    }
    fetchFileSamples();
  }, [dirPath, fileType, useDirectories]);

  const onFormSubmit = async (close: () => void) => {
    try {
      // Build samples array based on data schema type
      const samples: Partial<Sample>[] = shotIds.map((shotId) => {
        const baseSample = {
          shot_id: shotId,
          timestamp: new Date().toISOString(),
        };

        if (dataSchema?.title === "ShotData") {
          // Shot Data
          const signals = signalNames
            .split(",")
            .map((s) => s.trim())
            .filter((s) => s.length > 0);

          if (signals.length === 0) {
            throw new Error(
              "At least one signal name is required for Shot Data",
            );
          }

          return {
            ...baseSample,
            data: {
              protocol: "uda",
              signal_names: signals,
            } as ShotData,
          };
        } else if (
          dataSchema?.title === "ImageFileData" ||
          dataSchema?.title === "ImageArrayFileData" ||
          dataSchema?.title === "TimeSeriesFileData"
        ) {
          // File Data or Time Series File Data
          let fileName: string =
            dirPath + "/" + (shotId.toString() + "." + fileType); // Default file name format
          if (useDirectories) {
            fileName = dirPath + "/" + shotId.toString(); // For directories, the "file name" is just the directory path with shot ID
          }

          const fileData: Record<string, string | string[]> = {
            file_name: fileName,
            type: fileType,
            protocol: "file",
          };

          // Add column_names for time series file data
          if (columnNames.trim()) {
            const columns = columnNames
              .split(",")
              .map((s) => s.trim())
              .filter((s) => s.length > 0);

            if (
              dataSchema?.title === "TimeSeriesFileData" &&
              columns.length > 0
            ) {
              fileData.signal_names = columns;
              return {
                ...baseSample,
                data: fileData as TimeSeriesFileData,
              };
            } else if (
              dataSchema?.title === "ImageArrayFileData" &&
              columns.length > 1
            ) {
              throw new Error(
                "Only one signal name allowed for image array data!",
              );
            } else if (
              dataSchema?.title === "ImageArrayFileData" &&
              columns.length == 1
            ) {
              fileData.signal_name = columns[0];
              return {
                ...baseSample,
                data: fileData as ImageArrayFileData,
              };
            }
          }
          return {
            ...baseSample,
            data: fileData as FileData,
          };
        }

        return baseSample;
      });

      // POST to API
      const response = await apiFetch(
        `${BACKEND_API_URL}/projects/${project._id}/samples`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(samples),
        },
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Error creating samples");
      }

      const result = await response.json();
      ToastQueue.positive(
        `Successfully created ${result.length || samples.length} sample(s)!`,
        { timeout: 3000 },
      );

      if (onModify) {
        onModify();
      }

      close();
    } catch (error) {
      ToastQueue.negative(`${error}`, { timeout: 3000 });
    }
  };

  useEffect(() => {
    setShotIds([]);
  }, [dataSchema, shotInputMethod]);

  return (
    <DialogTrigger>
      <Button variant="primary">
        <AddCircle />
        <Text>Add Samples</Text>
      </Button>
      {(close) => (
        <Dialog>
          <Heading>Add Samples to Project</Heading>
          <Divider />
          <Content>
            {/* Display data type info from schema */}
            {dataSchema && (
              <Text>
                Data Type: <strong>{dataSchema.title}</strong> (from{" "}
                {dataLoader} data loader)
              </Text>
            )}

            {/* Conditional fields based on data schema type */}
            {dataSchema?.title === "ShotData" && (
              <Form maxWidth="size-6000">
                <TextField
                  label="Signal Names"
                  isRequired
                  value={signalNames}
                  onChange={setSignalNames}
                  description="Signal names to load as a comma-separated list, e.g., ip, dalpha, ANE_DENSITY"
                />
                <RadioGroup
                  label="Shot ID Input Method"
                  value={shotInputMethod}
                  onChange={setShotInputMethod}
                >
                  <Radio value="range">Numerical Range</Radio>
                  <Radio value="file">Text File</Radio>
                </RadioGroup>

                {shotInputMethod === "range" ? (
                  <NumericalRange
                    label="Shot"
                    isRequired
                    onChange={setShotRange}
                    maximumFractionDigits={0}
                    rangeMin={0}
                  />
                ) : (
                  <></>
                )}

                {shotInputMethod === "file" ? (
                  <Flex direction="column" gap="size-100">
                    <input
                      type="file"
                      accept=".txt,.csv"
                      onChange={(e) => handleFileSelect(e.target.files)}
                      style={{
                        padding: "8px",
                        border: "1px solid #ccc",
                        borderRadius: "4px",
                      }}
                    />
                    <Text>
                      Upload a CSV file with shot IDs in the first column
                      (header row will be skipped)
                    </Text>
                    {shotIds.length > 0 && (
                      <Text>
                        Loaded <strong>{shotIds.length}</strong> shot IDs:{" "}
                        {shotIds.slice(0, 5).join(", ")}
                        {shotIds.length > 5 &&
                          ` ... and ${shotIds.length - 5} more`}
                      </Text>
                    )}
                  </Flex>
                ) : (
                  <></>
                )}
              </Form>
            )}

            {(dataSchema?.title === "ImageFileData" ||
              dataSchema?.title === "ImageArrayFileData" ||
              dataSchema?.title === "TimeSeriesFileData") && (
              <Form maxWidth="size-6000">
                <ComboBox
                  label="File Type"
                  items={fileTypes}
                  isRequired
                  selectedKey={fileType}
                  onSelectionChange={(key) =>
                    setFileType(key ? String(key) : "parquet")
                  }
                  description="File extension to filter for. For directory paths, only files with this extension will be included."
                >
                  {(item: Record<string, string>) => (
                    <Item key={item.key}>{item.value}</Item>
                  )}
                </ComboBox>

                <Flex direction="row" gap="size-200" alignItems="end">
                  <TextField
                    label={"Directory Path"}
                    isRequired
                    flex={1}
                    value={dirPath}
                    onChange={setDirPath}
                    description={"Path to directory containing data files"}
                  />
                </Flex>
              </Form>
            )}

            {/* Display found shot IDs */}
            {(dataSchema?.title === "ImageFileData" ||
              dataSchema?.title === "ImageArrayFileData" ||
              dataSchema?.title === "TimeSeriesFileData") &&
            shotIds.length > 0 ? (
              <Text>
                Found <strong>{shotIds.length}</strong>{" "}
                {useDirectories ? "directories" : "files"} with shot IDs:{" "}
                {shotIds.slice(0, 5).join(", ")}
                {shotIds.length > 5 && ` ... and ${shotIds.length - 5} more`}
              </Text>
            ) : (
              <Text>
                No files found for the specified directory and file type.
              </Text>
            )}

            {(dataSchema?.title === "TimeSeriesFileData" ||
              dataSchema?.title === "ImageArrayFileData") && (
              <Form maxWidth="size-6000">
                <TextField
                  label={
                    dataSchema?.title === "TimeSeriesFileData"
                      ? "Signal Names (comma-separated)"
                      : "Signal Name"
                  }
                  value={columnNames}
                  onChange={setColumnNames}
                  description="Optional: Specify signal names to load data for."
                />
              </Form>
            )}
          </Content>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Cancel
            </Button>
            <Button variant="primary" onPress={async () => onFormSubmit(close)}>
              Add Samples
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
};
