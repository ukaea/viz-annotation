import { importJSONFile } from "@/app/core";
import { Project, Sample } from "@/types";
import {
  Button,
  FileTrigger,
  ToastQueue,
  Text,
} from "@adobe/react-spectrum";
import Import from "@spectrum-icons/workflow/Import";

export function ImportButton({
  project,
  sample,
  refreshAnnotations,
  canAnnotate = true,
}: {
  project: Project;
  sample?: Sample;
  refreshAnnotations?: () => void;
  canAnnotate?: boolean;
}) {
  const project_id = project._id;
  const shot_id = sample?.shot_id || null;

  if (!project_id) {
    return null;
  }

  const handleFileChange = (files: FileList | null) => {
    if (files && files.length > 0) {
      const file = files[0];
      importJSONFile(
        project_id,
        shot_id,
        file,
        () => {
          ToastQueue.positive(
            `Annotations imported successfully from ${file.name}`,
          );
          refreshAnnotations?.();
        },
        () => {
          ToastQueue.negative(`Error importing annotations from ${file.name}`);
        },
      );
    }
  };

  return (
    <FileTrigger
      onSelect={handleFileChange}
      acceptedFileTypes={["application/json"]}
    >
      <Button variant="primary" isDisabled={!canAnnotate}>
        <Import />
        <Text>{sample ? "Import" : "Import Annotations"}</Text>
      </Button>
    </FileTrigger>
  );
}
