# User Management

TokTagger supports multiple concurrent users with role-based access control. An **admin** user manages accounts and project membership; regular **users** annotate within the projects they are assigned to.

---

## User Roles

TokTagger has two layers of roles:

### Global roles (account-level)

| Global Role | Permissions |
|---|---|
| `admin` | Full access: create/edit/delete any project, manage all user accounts, view all annotations |
| `user` | Access only to projects they are a member of |

### Project roles (per-project membership)

| Project Role | Permissions |
|---|---|
| `admin` | Manage project membership, delete samples and annotations |
| `annotator` | Submit and update annotations for samples |
| `viewer` | Read-only access to the project's samples and annotations |

A global `admin` automatically has unrestricted access to all projects regardless of project role.

---

## First-Run Setup

On first launch TokTagger automatically creates an `admin` account with a fixed default password and prints the credentials to the terminal:

```
Username : admin
Password : admin
```

!!! warning
    This is an insecure default password. Change it immediately after first login from the **Profile** page.

---

## Signing In

Navigate to `http://<host>:<port>/ui/login` (or the root URL, which redirects there automatically). Enter your username and password to sign in.

---

## Admin Panel

The admin panel is accessible from the **Admin Panel** button on the Projects page (visible to admin users only).

### Viewing Users

The panel lists all registered accounts with their username, role, and active status.

### Creating a User

1. Click **Add User**.
2. Fill in **Username** and **Password**.
3. Select a **Role** (`user` or `admin`).
4. Click **Create**.

!!! warning
    You set the initial password. Give it to the new user through a secure channel. Do not send it by email or chat. TokTagger asks the new user to change the password at their next login.

### Resetting a User's Password

1. Find the user in the table and click **Reset Password**.
2. Type a new password.
3. Click **Reset**.

TokTagger asks the user to change this password at their next login. Give the new password to the user through a secure channel.

### Changing a User's Role

1. Find the user in the table and click **Edit**.
2. Select the new **Global Role**.
3. Click **Save**.

!!! note
    TokTagger prevents demoting or deactivating the last remaining active admin account to avoid an unrecoverable lockout.

### Deactivating / Reactivating a User

Click **Deactivate** (or **Activate**) next to the user. Deactivated accounts cannot sign in but their annotations are preserved. You cannot deactivate your own account.

### Deleting a User

Click **Delete** next to the user and confirm. This is permanent. You cannot delete your own account.

---

## Profile Page

Any signed-in user can update their own profile. Click **Profile** from the Projects page.

### Changing Password

1. Enter a new password in **New password** (minimum 8 characters).
2. Confirm it in **Confirm new password**.
3. Click **Change Password**.

---

## Project Membership

Access to a project is controlled per-project. From the project's Samples page, an admin can click **Members** to add or remove users.

Only members (and admins) can view samples and submit annotations for a given project.

---

## Scripted User & Project Setup

For automated deployments, the helper script `scripts/setup.py` can create projects and samples via the API using token-based auth:

```sh
python scripts/setup.py \
  --url http://localhost:8002 \
  --username admin \
  --password <password>
```

The script authenticates, obtains a JWT token, and creates projects and sample sets using the REST API. You can adapt it to pre-create user accounts with the `POST /users` endpoint:

```python
import requests

token = get_token(base_url, "admin", admin_password)
requests.post(
    f"{base_url}/users",
    json={"username": "alice", "password": "s3cr3t", "global_role": "user"},
    headers={"Authorization": f"Bearer {token}"},
)
```

---

## Multi-User Deployment

For team use, run the API under **Gunicorn** so multiple requests can be served concurrently:

```sh
# Command-line (installed package)
toktagger --workers 4 --host 0.0.0.0 --port 8002

# Direct Gunicorn invocation
gunicorn toktagger.api.asgi:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8002
```

With Docker Compose the `SERVER_WORKERS` variable controls the worker count (default 4 in production, 1 in dev):

```sh
SERVER_WORKERS=8 docker compose up
```

A single Uvicorn worker (the default for `toktagger` without `--workers`) is sufficient for personal/local use but will serialise all requests, so concurrent annotators will experience latency under load.
