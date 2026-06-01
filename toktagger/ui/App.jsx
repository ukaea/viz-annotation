import "./src/app/globals.css";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { Provider, defaultTheme, ToastContainer } from "@adobe/react-spectrum";
import { useNavigate, useHref } from "react-router-dom";
import { APISchemaProvider } from "./src/app/contexts/apiSchema";
import { AuthProvider, useAuth } from "./src/app/contexts/AuthContext";
import { ServerHealthProvider } from "./src/app/contexts/healthContext";
import Projects from "./src/app/projects/page";
import ProjectView from "./src/app/projects/project_id/page";
import SampleView from "./src/app/projects/project_id/samples/sample_id/page";
import LoginPage from "./src/app/pages/login";
import AdminUsersPage from "./src/app/pages/admin/users";
import ProfilePage from "./src/app/pages/profile";

function SpectrumProvider({ children }) {
  const navigate = useNavigate();
  return (
    <Provider theme={defaultTheme} router={{ navigate, useHref }}>
      <ToastContainer placement="top" />
      {children}
    </Provider>
  );
}

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
  if (user.global_role !== "admin")
    return <Navigate to="/ui/projects/" replace />;
  return children;
}

function App() {
  return (
    <APISchemaProvider>
      <Router>
        <SpectrumProvider>
          <ServerHealthProvider>
            <AuthProvider>
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
                <Route
                  path="/ui/profile"
                  element={
                    <RequireAuth>
                      <ProfilePage />
                    </RequireAuth>
                  }
                />
              </Routes>
            </AuthProvider>
          </ServerHealthProvider>
        </SpectrumProvider>
      </Router>
    </APISchemaProvider>
  );
}

export default App;
