import {
  Text,
  Button,
  ActionButton,
  Flex,
  Item,
  ComboBox,
  ContextualHelp,
  Content,
} from "@adobe/react-spectrum";
import { useState } from "react";
import Export from "@spectrum-icons/workflow/Export";
import { Project, Sample } from "@/types";
import { exportAnnotations } from "@/app/core";

export function ExportButton({ project }: { project: Project }) {
  return (
    <ActionButton isQuiet onPress={() => exportAnnotations(project)}>
      <Export />
      <Text>Export Annotations</Text>
    </ActionButton>
  );
}
type ExportToolInfo = {
  project: Project;
  sample: Sample;
};
export function ExportTool({ project, sample }: ExportToolInfo) {
  const exportItems = ["All", "Current Sample"].map((item) => ({
    id: item,
    name: item,
  }));
  const [exportOption, setExportOption] = useState<string>("All");

  const handleExport = () => {
    if (exportOption === "All") {
      exportAnnotations(project);
    } else {
      exportAnnotations(project, sample);
    }
  };

  return (
    <div className="m-4">
      <Flex direction="column">
        <Flex direction="row" alignItems="center" gap="size-100">
          <ComboBox
            label="Export"
            defaultSelectedKey={exportOption}
            defaultItems={exportItems}
            onInputChange={setExportOption}
          >
            {(item: { id: string; name: string }) => <Item>{item.name}</Item>}
          </ComboBox>
          <ContextualHelp placement="top">
            <Content>
              <Text>
                Export annotations for either all samples in the project or the
                current sample being displayed.
              </Text>
            </Content>
          </ContextualHelp>
        </Flex>
        <br />
        <Flex justifyContent="center">
          <Button variant="primary" onPress={handleExport}>
            <Export />
            <Text>Export</Text>
          </Button>
        </Flex>
      </Flex>
    </div>
  );
}
