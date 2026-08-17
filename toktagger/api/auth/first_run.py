from pathlib import Path

from filelock import FileLock

from toktagger.api import config
from toktagger.api.auth.core import hash_password
from toktagger.api.schemas.users import UserIn


async def ensure_admin_user(db_client) -> bool:
    """Create the default admin user on first run.

    Returns True if auth is required (users exist after this call).
    """
    lock_path = Path(config.settings.server.cache_dir) / "first_run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=30):
        users = await db_client.get_all_documents("users")
        if users:
            return True

        password = "admin"
        # The default password is public knowledge, so this account is no different
        # from an admin-created one: the first person to sign in must replace it
        # before they can reach any other page.
        admin = UserIn(
            username="admin",
            hashed_password=hash_password(password),
            global_role="admin",
            is_active=True,
            must_change_password=True,
        )
        await db_client.insert(collection="users", model=admin)

    border = "=" * 52
    print(f"\n{border}")
    print("  TokTagger: first-run setup")
    print("  Admin account created")
    print("  Username : admin")
    print(f"  Password : {password}")
    print("  ⚠  This is an insecure default password.")
    print("  ⚠  You must change it at first login.")
    print(f"{border}\n")

    return True
