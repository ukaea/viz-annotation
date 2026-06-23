import secrets
from pathlib import Path
from filelock import FileLock
from platformdirs import user_cache_dir
from toktagger.api.auth.core import hash_password
from toktagger.api.schemas.users import UserIn


async def ensure_admin_user(db_client) -> bool:
    """Create the default admin user on first run.

    Returns True if auth is required (users exist after this call).
    """
    lock_path = Path(user_cache_dir("toktagger", "ukaea")) / "first_run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=30):
        users = await db_client.get_all_documents("users")
        if users:
            return True

        password = secrets.token_urlsafe(12)
        admin = UserIn(
            username="admin",
            hashed_password=hash_password(password),
            email="",
            global_role="admin",
            is_active=True,
        )
        await db_client.insert(collection="users", model=admin)

    border = "=" * 52
    print(f"\n{border}")
    print("  TokTagger: first-run setup")
    print("  Admin account created")
    print("  Username : admin")
    print(f"  Password : {password}")
    print("  ⚠  This password will not be shown again.")
    print("  Please change it after first login.")
    print(f"{border}\n")

    return True
