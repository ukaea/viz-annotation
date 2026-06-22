"use client";
import { useState, useEffect, useCallback } from "react";
import {
  Button,
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
  InlineAlert,
  ToastQueue,
} from "@adobe/react-spectrum";
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
    await apiFetch(
      `${BACKEND_API_URL}/projects/${projectId}/members/${userId}`,
      { method: "DELETE" },
    );
    await refresh();
    ToastQueue.positive("Member removed", { timeout: 2000 });
  };

  const updateRole = async (userId: string, role: string) => {
    await apiFetch(
      `${BACKEND_API_URL}/projects/${projectId}/members/${userId}`,
      {
        method: "PUT",
        body: JSON.stringify({ role }),
      },
    );
    await refresh();
    ToastQueue.positive("Role updated", { timeout: 2000 });
  };

  return (
    <DialogTrigger isOpen={open} onOpenChange={setOpen}>
      <Button variant="secondary">Manage Members</Button>
      <Dialog width="size-6000">
        <Heading>Project Members</Heading>
        <Divider />
        <Content>
          {isProjectAdmin && (
            <AddMemberForm projectId={projectId} onAdded={refresh} />
          )}
          <TableView
            aria-label="Members"
            selectionMode="none"
            UNSAFE_style={{ marginTop: 16 }}
          >
            <TableHeader>
              <Column key="username">Username</Column>
              <Column key="role">Role</Column>
              {isProjectAdmin ? (
                <Column key="actions">Actions</Column>
              ) : (
                <Column key="placeholder"> </Column>
              )}
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
                    {isProjectAdmin ? (
                      <Button
                        variant="negative"
                        onPress={() => removeMember(item.user_id)}
                      >
                        Remove
                      </Button>
                    ) : (
                      ""
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
  const [role, setRole] = useState("annotator");
  const [error, setError] = useState<string | null>(null);

  const add = async () => {
    setError(null);
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
      setError(e instanceof Error ? e.message : "Error");
    }
  };

  return (
    <Flex gap="size-100" alignItems="end" wrap>
      {error && (
        <InlineAlert variant="negative" width="100%">
          {error}
        </InlineAlert>
      )}
      <TextField
        label="Username"
        value={username}
        onChange={setUsername}
        width="size-2400"
      />
      <Picker
        label="Role"
        selectedKey={role}
        onSelectionChange={(k) => setRole(k as string)}
        width="size-1600"
      >
        <Item key="admin">Admin</Item>
        <Item key="annotator">Annotator</Item>
        <Item key="viewer">Viewer</Item>
      </Picker>
      <Button
        variant="cta"
        isDisabled={!username}
        onPress={add}
        marginTop="size-200"
      >
        Add
      </Button>
    </Flex>
  );
}
