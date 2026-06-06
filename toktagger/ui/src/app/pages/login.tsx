"use client";
import { useState } from "react";
import { Navigate } from "react-router-dom";
import { Heading, InlineAlert, TextField, Button } from "@adobe/react-spectrum";
import { useAuth } from "@/app/contexts/AuthContext";

export default function LoginPage() {
  const { login, isLoading, user } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!isLoading && user) {
    return <Navigate to="/ui/projects/" replace />;
  }

  const handleLogin = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleLogin();
  };

  return (
    <div className="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400 dark:from-gray-700 dark:via-gray-800 dark:to-gray-900">
      <div
        className="bg-white dark:bg-gray-800 p-10 rounded-2xl shadow-2xl"
        style={{ minWidth: 360 }}
      >
        <Heading
          level={2}
          UNSAFE_style={{ marginBottom: 24, textAlign: "center" }}
        >
          TokTagger — Sign In
        </Heading>
        {error && (
          <InlineAlert
            variant="negative"
            UNSAFE_style={{ marginBottom: 16, width: "100%" }}
          >
            {error}
          </InlineAlert>
        )}
        <form
          onSubmit={handleSubmit}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 16,
            width: "100%",
          }}
        >
          <button
            type="submit"
            style={{ display: "none" }}
            tabIndex={-1}
            aria-hidden
          />
          <TextField
            label="Username"
            value={username}
            onChange={setUsername}
            isRequired
            autoFocus
            width="100%"
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={setPassword}
            isRequired
            width="100%"
          />
          <Button
            type="button"
            variant="cta"
            isPending={submitting}
            isDisabled={submitting || !username || !password}
            onPress={handleLogin}
            width="100%"
            UNSAFE_style={{ marginTop: 8 }}
          >
            Sign In
          </Button>
        </form>
      </div>
    </div>
  );
}
