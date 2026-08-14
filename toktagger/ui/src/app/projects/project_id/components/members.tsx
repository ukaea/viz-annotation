"use client";
import { useState, useEffect, useCallback } from "react";
import {
  Button,
  ActionButton,
  DialogTrigger,
  Dialog,
  Heading,
  Divider,
  Content,
  ButtonGroup,
  TableView,
  TableHeader,
  TableBody,
  Column,
  Row,
  Cell,
  Flex,
  TextField,
  Picker,
  Item,
  Text,
  ToastQueue,
} from "@adobe/react-spectrum";
import UserGroup from "@spectrum-icons/workflow/UserGroup";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import type { ProjectMember } from "@/types";

interface Props {
  projectId: string;
  isProjectAdmin: boolean;
}

export function ProjectMembersDialog({ projectId, isProjectAdmin }: Props) {
  const [members, setMembers] = useState<ProjectMember[]>([]);
  const [open, setOpen] = useState(false);

  const refresh = useCallback(async () => {
    const res = await apiFetch(
      `${BACKEND_API_URL}/projects/${projectId}/members`,
    );
    if (res.ok) {
      setMembers(await res.json());
    }
  }, [projectId]);

  useEffect(() => {
    if (open) refresh();
  }, [open, refresh]);

  const removeMember = async (userId: string) => {
    try {
      const res = await apiFetch(
        `${BACKEND_API_URL}/projects/${projectId}/members/${userId}`,
        { method: "DELETE" },
      );
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? "Failed to remove member");
      }
      await refresh();
      ToastQueue.positive("Member removed", { timeout: 2000 });
    } catch (e) {
      ToastQueue.negative(e instanceof Error ? e.message : "Error", {
        timeout: 3000,
      });
    }
  };

  const updateRole = async (userId: string, role: string) => {
    try {
      const res = await apiFetch(
        `${BACKEND_API_URL}/projects/${projectId}/members/${userId}`,
        {
          method: "PUT",
          body: JSON.stringify({ role }),
        },
      );
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? "Failed to update role");
      }
      await refresh();
      ToastQueue.positive("Role updated", { timeout: 2000 });
    } catch (e) {
      ToastQueue.negative(e instanceof Error ? e.message : "Error", {
        timeout: 3000,
      });
    }
  };

  return (
    <DialogTrigger isOpen={open} onOpenChange={setOpen}>
      <ActionButton isQuiet>
        <UserGroup />
        <Text>Manage Members</Text>
      </ActionButton>
      <Dialog width="size-8000">
        <Heading>Project Members</Heading>
        <Divider />
        <Content>
          {isProjectAdmin && (
            <AddMemberForm projectId={projectId} onAdded={refresh} />
          )}
          <TableView
            aria-label="Members"
            selectionMode="none"
            marginTop="size-200"
          >
            <TableHeader>
              <Column key="username">Username</Column>
              <Column key="role">Role</Column>
              <Column key="actions">{isProjectAdmin ? "Actions" : ""}</Column>
            </TableHeader>
            <TableBody items={members}>
              {(item) => (
                <Row key={item._id}>
                  <Cell>{item.username}</Cell>
                  <Cell>
                    {isProjectAdmin ? (
                      <Picker
                        aria-label="Role"
                        selectedKey={item.role}
                        onSelectionChange={(k) =>
                          updateRole(item.user_id, k as string)
                        }
                        width="size-1600"
                      >
                        <Item key="admin">Admin</Item>
                        <Item key="annotator">Annotator</Item>
                        <Item key="viewer">Viewer</Item>
                      </Picker>
                    ) : (
                      item.role
                    )}
                  </Cell>
                  <Cell>
                    {isProjectAdmin && (
                      <Button
                        variant="negative"
                        onPress={() => removeMember(item.user_id)}
                      >
                        Remove
                      </Button>
                    )}
                  </Cell>
                </Row>
              )}
            </TableBody>
          </TableView>
        </Content>
        <ButtonGroup>
          <Button variant="secondary" onPress={() => setOpen(false)}>
            Close
          </Button>
        </ButtonGroup>
      </Dialog>
    </DialogTrigger>
  );
}

function AddMemberForm({
  projectId,
  onAdded,
}: {
  projectId: string;
  onAdded: () => void;
}) {
  const [username, setUsername] = useState("");
  const [role, setRole] = useState<"admin" | "annotator" | "viewer">(
    "annotator",
  );

  const add = async () => {
    try {
      const res = await apiFetch(
        `${BACKEND_API_URL}/projects/${projectId}/members`,
        {
          method: "POST",
          body: JSON.stringify({ username, role }),
        },
      );
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? "Failed to add member");
      }
      setUsername("");
      setRole("annotator");
      onAdded();
      ToastQueue.positive("Member added", { timeout: 2000 });
    } catch (e) {
      ToastQueue.negative(e instanceof Error ? e.message : "Error", {
        timeout: 3000,
      });
    }
  };

  return (
    <Flex gap="size-100" alignItems="end" wrap>
      <TextField
        label="Username"
        value={username}
        onChange={setUsername}
        width="size-2400"
      />
      <Picker
        label="Role"
        selectedKey={role}
        onSelectionChange={(k) =>
          setRole(k as "admin" | "annotator" | "viewer")
        }
        width="size-1600"
      >
        <Item key="admin">Admin</Item>
        <Item key="annotator">Annotator</Item>
        <Item key="viewer">Viewer</Item>
      </Picker>
      <Button variant="cta" isDisabled={!username} onPress={add}>
        Add
      </Button>
    </Flex>
  );
}
