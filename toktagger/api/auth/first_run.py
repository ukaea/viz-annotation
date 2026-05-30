import secrets
from toktagger.api.auth.core import hash_password
from toktagger.api.schemas.users import UserIn


async def ensure_admin_user(db_client) -> bool:
    """Create the default admin user on first run.

    Returns True if auth is required (users exist after this call).
    """
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
    print(f"  TokTagger: first-run setup")
    print(f"  Admin account created")
    print(f"  Username : admin")
    print(f"  Password : {password}")
    print(f"  Please change this password after first login.")
    print(f"{border}\n")

    return True
