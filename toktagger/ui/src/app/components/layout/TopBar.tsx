"use client";
import { Breadcrumbs, Item, Button } from "@adobe/react-spectrum";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/app/contexts/AuthContext";
import { useBreadcrumbItems } from "@/app/contexts/BreadcrumbContext";

export default function TopBar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const isAdmin = user?.global_role === "admin";
  const breadcrumbItems = useBreadcrumbItems();

  return (
    <div className="w-full flex-none h-14 flex items-center justify-between gap-4 px-6 border-b border-gray-300 dark:border-gray-700 bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm text-gray-800 dark:text-gray-100">
      {/* flex-1 so the breadcrumbs get the space the right-hand buttons leave over.
          Without it this div shrinks to its content-free width and Spectrum folds
          every crumb but the last into a "…" menu. min-w-0 keeps it from pushing
          the buttons off the bar when a project name is long. */}
      <div className="min-w-0 flex-1">
        {breadcrumbItems.length > 0 && (
          <Breadcrumbs>
            {breadcrumbItems.map((item) => (
              <Item key={item.key} href={item.href}>
                {item.label}
              </Item>
            ))}
          </Breadcrumbs>
        )}
      </div>
      <div className="flex items-center gap-2 flex-none">
        <span className="text-sm text-gray-600 dark:text-gray-300">
          Signed in as <strong>{user?.username}</strong>
        </span>
        <Button variant="secondary" onPress={() => navigate("/ui/profile")}>
          Profile
        </Button>
        {isAdmin && (
          <Button
            variant="secondary"
            onPress={() => navigate("/ui/admin/users")}
          >
            Admin Panel
          </Button>
        )}
        <Button variant="negative" onPress={logout}>
          Sign Out
        </Button>
      </div>
    </div>
  );
}
