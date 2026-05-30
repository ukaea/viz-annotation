import "./src/app/globals.css";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { APISchemaProvider } from "./src/app/contexts/apiSchema";
import { AuthProvider, useAuth } from "./src/app/contexts/AuthContext";
import { ServerHealthProvider } from "@/app/contexts/healthContext";
import Projects from "./src/app/projects/page";
import ProjectView from "./src/app/projects/project_id/page";
import SampleView from "./src/app/projects/project_id/samples/sample_id/page";
import LoginPage from "./src/app/pages/login";
import AdminUsersPage from "./src/app/pages/admin/users";

function RequireAuth({ children }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  if (!user) return <Navigate to="/ui/login" replace />;
  return children;
}

function RequireAdmin({ children }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  if (!user) return <Navigate to="/ui/login" replace />;
  if (user.global_role !== "admin") return <Navigate to="/ui/projects/" replace />;
  return children;
}

function App() {
  return (
    <APISchemaProvider>
      <Router>
        <AuthProvider>
          <ServerHealthProvider>
            <Routes>
              <Route path="/ui/login" element={<LoginPage />} />
              <Route
                path="/"
                element={
                  <RequireAuth>
                    <Projects />
                  </RequireAuth>
                }
              />
              <Route
                path="/ui/projects/"
                element={
                  <RequireAuth>
                    <Projects />
                  </RequireAuth>
                }
              />
              <Route
                path="/ui/projects/:project_id"
                element={
                  <RequireAuth>
                    <ProjectView />
                  </RequireAuth>
                }
              />
              <Route
                path="/ui/projects/:project_id/samples/:sample_id"
                element={
                  <RequireAuth>
                    <SampleView />
                  </RequireAuth>
                }
              />
              <Route
                path="/ui/admin/users"
                element={
                  <RequireAdmin>
                    <AdminUsersPage />
                  </RequireAdmin>
                }
              />
            </Routes>
          </ServerHealthProvider>
        </AuthProvider>
      </Router>
    </APISchemaProvider>
  );
}

export default App;
