"use client";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Provider,
  defaultTheme,
  Form,
  TextField,
  Button,
  Heading,
  View,
  InlineAlert,
} from "@adobe/react-spectrum";
import { useAuth } from "@/app/contexts/AuthContext";

export default function LoginPage() {
  const { login, isLoading, user } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Already logged in — redirect
  if (!isLoading && user) {
    navigate("/ui/projects/");
    return null;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
      navigate("/ui/projects/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Provider theme={defaultTheme}>
      <div className="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400">
        <View
          backgroundColor="static-white"
          padding="size-400"
          borderRadius="medium"
          UNSAFE_style={{ boxShadow: "0 4px 24px rgba(0,0,0,0.15)", minWidth: 360 }}
        >
          <Heading level={2} UNSAFE_style={{ marginBottom: 24 }}>
            TokTagger — Sign In
          </Heading>
          {error && (
            <InlineAlert variant="negative" UNSAFE_style={{ marginBottom: 16 }}>
              {error}
            </InlineAlert>
          )}
          <form onSubmit={handleSubmit}>
            <Form>
              <TextField
                label="Username"
                value={username}
                onChange={setUsername}
                isRequired
                autoFocus
              />
              <TextField
                label="Password"
                type="password"
                value={password}
                onChange={setPassword}
                isRequired
              />
              <Button
                type="submit"
                variant="cta"
                isPending={submitting}
                isDisabled={submitting || !username || !password}
                UNSAFE_style={{ marginTop: 16 }}
              >
                Sign In
              </Button>
            </Form>
          </form>
        </View>
      </div>
    </Provider>
  );
}
