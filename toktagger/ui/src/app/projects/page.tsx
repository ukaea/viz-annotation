"use client";
import { useState, useEffect, useCallback } from "react";
import { deleteProject, getProjects } from "@/app/core";
import Delete from "@spectrum-icons/workflow/Delete";
import { ProjectConfigEditor } from "./components/project_config";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/app/contexts/AuthContext";
import {
  Cell,
  Column,
  Row,
  TableView,
  TableBody,
  TableHeader,
  Breadcrumbs,
  Item,
  Button,
  Picker,
  Flex,
  SearchField,
  DialogTrigger,
  Dialog,
  Divider,
  Heading,
  Content,
  ButtonGroup,
} from "@adobe/react-spectrum";
import type { SortDescriptor } from "@react-types/shared";
import type { Project } from "@/types";

type ProjectsTableProps = {
  projects: Project[];
  sortDescriptor: SortDescriptor;
  onSortChange: (sort: SortDescriptor) => void;
  onModify?: () => void;
};

const ProjectsTable = ({ projects, sortDescriptor, onSortChange, onModify }: ProjectsTableProps) => {
  const rows = projects.map(({ _id, ...rest }) => ({ ...rest, id: _id, _id }));

  return (
    <Flex height="size-5000" width="100%" direction="column">
      <TableView
        flex
        aria-label="Projects"
        selectionMode="none"
        selectionStyle="highlight"
        sortDescriptor={sortDescriptor}
        onSortChange={onSortChange}
      >
        <TableHeader>
          <Column key="name" allowsSorting>Name</Column>
          <Column key="task" allowsSorting>Task</Column>
          <Column key="timestamp" allowsSorting>Date Created</Column>
          <Column key="data_loader" allowsSorting>Loader</Column>
          <Column key="actions">Actions</Column>
        </TableHeader>
        <TableBody items={rows}>
          {(item) => (
            <Row href={`/ui/projects/${item._id}`}>
              <Cell>{item["name"]}</Cell>
              <Cell>{item["task"]}</Cell>
              <Cell>{item["timestamp"]}</Cell>
              <Cell>{item["data_loader"]}</Cell>
              <Cell>
                <Flex direction="row" gap="size-100">
                  <ProjectConfigEditor project={item} onModify={onModify} />
                  <DialogTrigger>
                    <Button aria-label="Delete" variant="negative"><Delete /></Button>
                    {(close) => (
                      <Dialog>
                        <Heading>Confirm Deletion</Heading>
                        <Divider />
                        <Content>
                          Are you sure you want to delete project <strong>{item["name"]}</strong>?
                          You will also lose <strong>all annotations</strong> associated with this
                          project. This action cannot be undone.
                        </Content>
                        <ButtonGroup>
                          <Button variant="secondary" onPress={close}>Cancel</Button>
                          <Button variant="negative" onPress={async () => { await deleteProject(item._id); onModify?.(); close(); }}>
                            Delete
                          </Button>
                        </ButtonGroup>
                      </Dialog>
                    )}
                  </DialogTrigger>
                </Flex>
              </Cell>
            </Row>
          )}
        </TableBody>
      </TableView>
    </Flex>
  );
};

export default function Projects() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [projectsPerPage, setProjectsPerPage] = useState<number>(10);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [projectName, setProjectName] = useState<string>("");
  const [sortDescriptor, setSortDescriptor] = useState<SortDescriptor>({ column: "_id", direction: "descending" });
  const [projects, setProjects] = useState<Project[]>([]);

  const refreshProjects = useCallback(async () => {
    setProjects(await getProjects(sortDescriptor, currentPage, projectsPerPage, projectName));
  }, [sortDescriptor, currentPage, projectsPerPage, projectName]);

  useEffect(() => { refreshProjects(); }, [refreshProjects]);

  if (!projects) return;

  return (
    <div>
      <Breadcrumbs>
        <Item key="projects" href="/ui/projects/">Projects</Item>
      </Breadcrumbs>
      <div className="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400 dark:from-gray-700 dark:via-gray-800 dark:to-gray-900">
        <div className="w-full md:w-4/5 p-6 bg-white/60 dark:bg-gray-800/60 text-gray-800 dark:text-gray-100 rounded-lg shadow-lg backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-bold">Projects</h1>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600 dark:text-gray-300">
                Signed in as <strong>{user?.username}</strong>
              </span>
              <Button variant="secondary" onPress={() => navigate("/ui/profile")}>Profile</Button>
              {user?.global_role === "admin" && (
                <Button variant="secondary" onPress={() => navigate("/ui/admin/users")}>Admin Panel</Button>
              )}
              <Button variant="negative" onPress={logout}>Sign Out</Button>
            </div>
          </div>
          <Flex direction="row" margin="size-100" gap="size-100" alignItems="center" justifyContent="space-between">
            <ProjectConfigEditor onModify={refreshProjects} />
            <SearchField
              label="Search By Name"
              onSubmit={(name) => { if (name != null) { setProjectName(name); setCurrentPage(1); } }}
            />
          </Flex>
          <ProjectsTable
            projects={projects}
            sortDescriptor={sortDescriptor}
            onSortChange={(d) => setSortDescriptor(d)}
            onModify={refreshProjects}
          />
          <div className="flex items-center justify-between pl-4 pr-4">
            <Button variant="primary" onPress={() => setCurrentPage((p) => p - 1)} isDisabled={currentPage === 1}>
              Previous
            </Button>
            <div className="flex items-center justify-center gap-8 pb-2">
              <p>Page: {currentPage}</p>
              <Picker
                label="Projects per Page:"
                onSelectionChange={(k) => { if (k != null) { setProjectsPerPage(Number(k) || 10); setCurrentPage(1); } }}
                defaultSelectedKey="10"
              >
                <Item key="5">5</Item>
                <Item key="10">10</Item>
                <Item key="25">25</Item>
                <Item key="50">50</Item>
              </Picker>
            </div>
            <Button variant="primary" onPress={() => setCurrentPage((p) => p + 1)} isDisabled={projects.length < projectsPerPage}>
              Next
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
