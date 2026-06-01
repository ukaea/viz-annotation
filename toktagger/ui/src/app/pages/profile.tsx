"use client";
import { useState } from "react";
import {
  Breadcrumbs,
  Item,
  Divider,
  TextField,
  Button,
  Flex,
  ToastQueue,
} from "@adobe/react-spectrum";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { useAuth } from "@/app/contexts/AuthContext";

export default function ProfilePage() {
  const { user } = useAuth();

  const [email, setEmail] = useState(user?.email ?? "");
  const [emailSaving, setEmailSaving] = useState(false);

  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordSaving, setPasswordSaving] = useState(false);

  const saveEmail = async () => {
    if (!user) return;
    setEmailSaving(true);
    try {
      const res = await apiFetch(`${BACKEND_API_URL}/users/${user._id}`, {
        method: "PUT",
        body: JSON.stringify({ email }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? "Failed to save email");
      }
      ToastQueue.positive("Email updated", { timeout: 2000 });
    } catch (e) {
      ToastQueue.negative(e instanceof Error ? e.message : "Error", { timeout: 3000 });
    } finally {
      setEmailSaving(false);
    }
  };

  const savePassword = async () => {
    if (!user) return;
    if (newPassword !== confirmPassword) {
      ToastQueue.negative("Passwords do not match", { timeout: 3000 });
      return;
    }
    if (newPassword.length < 8) {
      ToastQueue.negative("Password must be at least 8 characters", { timeout: 3000 });
      return;
    }
    setPasswordSaving(true);
    try {
      const res = await apiFetch(`${BACKEND_API_URL}/users/${user._id}`, {
        method: "PUT",
        body: JSON.stringify({ password: newPassword }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? "Failed to change password");
      }
      ToastQueue.positive("Password changed", { timeout: 2000 });
      setNewPassword("");
      setConfirmPassword("");
    } catch (e) {
      ToastQueue.negative(e instanceof Error ? e.message : "Error", { timeout: 3000 });
    } finally {
      setPasswordSaving(false);
    }
  };

  return (
    <div>
      <Breadcrumbs>
        <Item key="projects" href="/ui/projects/">Projects</Item>
        <Item key="profile">Profile</Item>
      </Breadcrumbs>
      <div className="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400 dark:from-gray-700 dark:via-gray-800 dark:to-gray-900">
        <div className="w-full md:w-4/5 p-6 bg-white/60 dark:bg-gray-800/60 text-gray-800 dark:text-gray-100 rounded-lg shadow-lg backdrop-blur-sm">
          <h1 className="text-2xl font-bold mb-4">Profile</h1>
          <Flex direction="column" alignItems="center">
            <Flex direction="column" gap="size-300" width="size-6000" maxWidth="100%">

              <p className="text-sm text-gray-600 dark:text-gray-300">
                <strong>Username:</strong> {user?.username}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>Role:</strong> {user?.global_role}
              </p>

              <TextField label="Email" value={email} onChange={setEmail} type="email" width="100%" />
              <Button variant="primary" onPress={saveEmail} isPending={emailSaving} isDisabled={emailSaving} width="100%">
                Save Email
              </Button>

              <Divider size="S" />

              <h2 className="text-lg font-semibold">Change Password</h2>
              <TextField label="New password" type="password" value={newPassword} onChange={setNewPassword} width="100%" />
              <TextField label="Confirm new password" type="password" value={confirmPassword} onChange={setConfirmPassword} width="100%" />
              <Button
                variant="primary"
                onPress={savePassword}
                isPending={passwordSaving}
                isDisabled={passwordSaving || !newPassword || !confirmPassword}
                width="100%"
              >
                Change Password
              </Button>

            </Flex>
          </Flex>
        </div>
      </div>
    </div>
  );
}
