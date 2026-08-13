"use client";
import { useEffect, useState } from "react";
import { BACKEND_API_URL, apiFetch } from "@/app/core";
import { useAuth } from "@/app/contexts/AuthContext";

type ProjectRole = "admin" | "annotator" | "viewer" | null;

export type ProjectRoleInfo = {
  // This project's membership role for the current user, or "admin" for a global
  // admin who bypasses membership entirely. null while loading or if the user has
  // no membership and isn't a global admin.
  role: ProjectRole;
  // Global admin, or a project-level admin - can manage members, delete samples/
  // annotations. Mirrors the backend's require_project_admin_role.
  isAdmin: boolean;
  // Global admin, or a project-level admin/annotator - can create/edit annotations
  // and samples. Mirrors the backend's require_project_annotator.
  canAnnotate: boolean;
  loading: boolean;
};

// Fetches the current user's membership for a project and derives the permission
// booleans every gated button/control needs. A global admin bypasses membership
// checks entirely, matching the backend's dependencies in api/auth/dependencies.py.
//
// isAdmin/canAnnotate default true and only correct downward once the membership
// check resolves. The backend is the real authority either way (these booleans only
// drive client-side disabling), and defaulting closed instead would briefly disable
// controls for a legitimate admin/annotator on every fresh mount - a window narrow in
// wall-clock time but long enough to swallow a click that fires before it clears.
export function useProjectRole(
  project_id: string | null | undefined,
): ProjectRoleInfo {
  const { user } = useAuth();
  const [role, setRole] = useState<ProjectRole>(null);
  const [restricted, setRestricted] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user || !project_id) {
      setLoading(false);
      return;
    }
    if (user.global_role === "admin") {
      setRole("admin");
      setRestricted(false);
      setLoading(false);
      return;
    }
    setLoading(true);
    apiFetch(`${BACKEND_API_URL}/projects/${project_id}/members`)
      .then((r) => r.json())
      .then((members: Array<{ user_id: string; role: ProjectRole }>) => {
        const membership = members.find((m) => m.user_id === user._id);
        setRole(membership?.role ?? null);
        setRestricted(true);
      })
      .catch(() => {
        setRole(null);
        setRestricted(true); // fail closed on a real error
      })
      .finally(() => setLoading(false));
  }, [project_id, user]);

  return {
    role,
    isAdmin: restricted ? role === "admin" : true,
    canAnnotate: restricted ? role === "admin" || role === "annotator" : true,
    loading,
  };
}
