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
  // Global admin, or a project-level admin - can manage members and delete a trained
  // model artifact. Mirrors the backend's require_project_admin_role.
  isAdmin: boolean;
  // Global admin, or a project-level admin/annotator - can create, edit and delete
  // annotations and samples, and edit or delete the project itself. Mirrors the
  // backend's require_project_annotator.
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

export type MyProjectRoles = {
  roleFor: (project_id: string | null | undefined) => ProjectRole;
  isAdminIn: (project_id: string | null | undefined) => boolean;
  canAnnotateIn: (project_id: string | null | undefined) => boolean;
  loading: boolean;
};

// The same permission booleans as useProjectRole, but for every project the user
// belongs to in a single request. The projects list gates each row, and mounting
// useProjectRole per row costs one /projects/{id}/members request per row.
//
// Unlike useProjectRole this reports "not permitted" while loading rather than
// defaulting open: the list only shows or hides whole controls, so failing open
// would flash Edit/Delete buttons on every row and then withdraw them. There is no
// click to swallow, because the buttons are simply not rendered yet. A global admin
// resolves synchronously from the auth context and never waits.
export function useMyProjectRoles(): MyProjectRoles {
  const { user } = useAuth();
  const [roles, setRoles] = useState<Record<string, ProjectRole>>({});
  const [loading, setLoading] = useState(true);

  const isGlobalAdmin = user?.global_role === "admin";

  useEffect(() => {
    if (!user || isGlobalAdmin) {
      setLoading(false);
      return;
    }
    setLoading(true);
    apiFetch(`${BACKEND_API_URL}/users/me/memberships`)
      .then((r) => r.json())
      .then((memberships: Array<{ project_id: string; role: ProjectRole }>) => {
        setRoles(
          Object.fromEntries(memberships.map((m) => [m.project_id, m.role])),
        );
      })
      .catch(() => setRoles({})) // fail closed on a real error
      .finally(() => setLoading(false));
  }, [user, isGlobalAdmin]);

  const roleFor = (project_id: string | null | undefined): ProjectRole => {
    if (isGlobalAdmin) return "admin";
    return project_id ? (roles[project_id] ?? null) : null;
  };

  return {
    roleFor,
    isAdminIn: (project_id) => roleFor(project_id) === "admin",
    canAnnotateIn: (project_id) => {
      const role = roleFor(project_id);
      return role === "admin" || role === "annotator";
    },
    loading,
  };
}
