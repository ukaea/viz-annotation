"use client";
import { useState, useEffect, useCallback } from "react";
import {
  TableView,
  TableHeader,
  TableBody,
  Column,
  Row,
  Cell,
  Button,
  Flex,
  DialogTrigger,
  Dialog,
  Heading,
  Divider,
  Content,
  ButtonGroup,
  TextField,
  Picker,
  Item,
  InlineAlert,
  ToastQueue,
  Breadcrumbs,
} from "@adobe/react-spectrum";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { useAuth } from "@/app/contexts/AuthContext";
import type { CurrentUser } from "@/types";

type UserRow = CurrentUser & { id: string };

export default function AdminUsersPage() {
  const { user: currentUser, logout } = useAuth();
  const [users, setUsers] = useState<UserRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await apiFetch(`${BACKEND_API_URL}/users`);
      if (!res.ok) throw new Error("Failed to load users");
      const data: CurrentUser[] = await res.json();
      setUsers(data.map((u) => ({ ...u, id: u._id })));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const deactivate = async (userId: string, isActive: boolean) => {
    await apiFetch(`${BACKEND_API_URL}/users/${userId}`, {
      method: "PUT",
      body: JSON.stringify({ is_active: !isActive }),
    });
    await refresh();
    ToastQueue.positive(isActive ? "User deactivated" : "User activated", {
      timeout: 2000,
    });
  };

  const deleteUser = async (userId: string, close: () => void) => {
    await apiFetch(`${BACKEND_API_URL}/users/${userId}`, { method: "DELETE" });
    close();
    await refresh();
    ToastQueue.positive("User deleted", { timeout: 2000 });
  };

  return (
    <div>
      <Breadcrumbs>
        <Item key="projects" href="/ui/projects/">
          Projects
        </Item>
        <Item key="admin">Admin</Item>
        <Item key="users">Users</Item>
      </Breadcrumbs>
      <div className="w-screen min-h-screen flex items-start justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400 dark:from-gray-700 dark:via-gray-800 dark:to-gray-900 py-6">
        <div className="w-full md:w-4/5 p-6 bg-white/60 dark:bg-gray-800/60 text-gray-800 dark:text-gray-100 rounded-lg shadow-lg backdrop-blur-sm">
          <Flex
            justifyContent="space-between"
            alignItems="center"
            marginBottom="size-200"
          >
            <Heading level={2}>User Management</Heading>
            <Flex gap="size-100" alignItems="center">
              <span className="text-sm text-gray-600 dark:text-gray-300">
                Signed in as <strong>{currentUser?.username}</strong>
              </span>
              <Button variant="negative" onPress={logout}>
                Sign Out
              </Button>
            </Flex>
          </Flex>

          {error && (
            <InlineAlert variant="negative" marginBottom="size-200">
              {error}
            </InlineAlert>
          )}

          <Flex marginBottom="size-200">
            <CreateUserDialog onCreated={refresh} />
          </Flex>

          <TableView aria-label="Users" selectionMode="none">
            <TableHeader>
              <Column key="username">Username</Column>
              <Column key="email">Email</Column>
              <Column key="global_role">Role</Column>
              <Column key="is_active">Active</Column>
              <Column key="actions">Actions</Column>
            </TableHeader>
            <TableBody items={users}>
              {(item) => (
                <Row key={item.id}>
                  <Cell>{item.username}</Cell>
                  <Cell>{item.email || "—"}</Cell>
                  <Cell>{item.global_role}</Cell>
                  <Cell>{item.is_active ? "Yes" : "No"}</Cell>
                  <Cell>
                    <Flex gap="size-100">
                      <ChangeRoleDialog user={item} onChanged={refresh} />
                      <Button
                        variant="secondary"
                        isDisabled={item.id === currentUser?._id}
                        onPress={() => deactivate(item.id, item.is_active)}
                      >
                        {item.is_active ? "Deactivate" : "Activate"}
                      </Button>
                      <DialogTrigger>
                        <Button
                          variant="negative"
                          isDisabled={item.id === currentUser?._id}
                        >
                          Delete
                        </Button>
                        {(close) => (
                          <Dialog>
                            <Heading>Delete User</Heading>
                            <Divider />
                            <Content>
                              Delete user <strong>{item.username}</strong>? This
                              cannot be undone.
                            </Content>
                            <ButtonGroup>
                              <Button variant="secondary" onPress={close}>
                                Cancel
                              </Button>
                              <Button
                                variant="negative"
                                onPress={() => deleteUser(item.id, close)}
                              >
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
        </div>
      </div>
    </div>
  );
}

function CreateUserDialog({ onCreated }: { onCreated: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"admin" | "user">("user");
  const [error, setError] = useState<string | null>(null);

  const submit = async (close: () => void) => {
    setError(null);
    try {
      const res = await apiFetch(`${BACKEND_API_URL}/users`, {
        method: "POST",
        body: JSON.stringify({ username, password, email, global_role: role }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? "Failed to create user");
      }
      setUsername("");
      setPassword("");
      setEmail("");
      setRole("user");
      close();
      onCreated();
      ToastQueue.positive("User created", { timeout: 2000 });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error");
    }
  };

  return (
    <DialogTrigger>
      <Button variant="cta">Add User</Button>
      {(close) => (
        <Dialog>
          <Heading>Create User</Heading>
          <Divider />
          <Content>
            {error && (
              <InlineAlert variant="negative" marginBottom="size-100">
                {error}
              </InlineAlert>
            )}
            <Flex direction="column" gap="size-100">
              <TextField
                label="Username"
                value={username}
                onChange={setUsername}
                isRequired
              />
              <TextField
                label="Password"
                type="password"
                value={password}
                onChange={setPassword}
                isRequired
              />
              <TextField label="Email" value={email} onChange={setEmail} />
              <Picker
                label="Role"
                selectedKey={role}
                onSelectionChange={(k) => setRole(k as "admin" | "user")}
              >
                <Item key="user">User</Item>
                <Item key="admin">Admin</Item>
              </Picker>
            </Flex>
          </Content>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Cancel
            </Button>
            <Button
              variant="cta"
              isDisabled={!username || !password}
              onPress={() => submit(close)}
            >
              Create
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}

function ChangeRoleDialog({
  user,
  onChanged,
}: {
  user: UserRow;
  onChanged: () => void;
}) {
  const [role, setRole] = useState<"admin" | "user">(user.global_role);

  const save = async (close: () => void) => {
    await apiFetch(`${BACKEND_API_URL}/users/${user.id}`, {
      method: "PUT",
      body: JSON.stringify({ global_role: role }),
    });
    close();
    onChanged();
    ToastQueue.positive("Role updated", { timeout: 2000 });
  };

  return (
    <DialogTrigger>
      <Button variant="secondary">Edit</Button>
      {(close) => (
        <Dialog>
          <Heading>Edit {user.username}</Heading>
          <Divider />
          <Content>
            <Picker
              label="Global Role"
              selectedKey={role}
              onSelectionChange={(k) => setRole(k as "admin" | "user")}
            >
              <Item key="user">User</Item>
              <Item key="admin">Admin</Item>
            </Picker>
          </Content>
          <ButtonGroup>
            <Button variant="secondary" onPress={close}>
              Cancel
            </Button>
            <Button variant="cta" onPress={() => save(close)}>
              Save
            </Button>
          </ButtonGroup>
        </Dialog>
      )}
    </DialogTrigger>
  );
}
