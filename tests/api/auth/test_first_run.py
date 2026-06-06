"""Unit tests for toktagger.api.auth.first_run."""

import pytest

from toktagger.api.auth.first_run import ensure_admin_user


@pytest.mark.asyncio
async def test_ensure_admin_user_creates_admin_on_empty_db(db_client):
    users_before = await db_client.get_all_documents("users")
    assert len(users_before) == 0

    await ensure_admin_user(db_client)

    users_after = await db_client.get_all_documents("users")
    assert len(users_after) == 1
    assert users_after[0]["username"] == "admin"
    assert users_after[0]["global_role"] == "admin"
    assert users_after[0]["is_active"] is True


@pytest.mark.asyncio
async def test_ensure_admin_user_returns_true(db_client):
    result = await ensure_admin_user(db_client)
    assert result is True


@pytest.mark.asyncio
async def test_ensure_admin_user_password_is_hashed(db_client):
    await ensure_admin_user(db_client)
    users = await db_client.get_all_documents("users")
    stored = users[0]["hashed_password"]
    # Must be stored in pbkdf2 format, not plain text
    assert stored.startswith("pbkdf2:")


@pytest.mark.asyncio
async def test_ensure_admin_user_idempotent(db_client):
    """Calling twice should not create a second admin."""
    await ensure_admin_user(db_client)
    await ensure_admin_user(db_client)

    users = await db_client.get_all_documents("users")
    assert len(users) == 1


@pytest.mark.asyncio
async def test_ensure_admin_user_with_existing_users_returns_true(db_client):
    """When users already exist, still returns True without creating another."""
    await ensure_admin_user(db_client)
    result = await ensure_admin_user(db_client)
    assert result is True

    users = await db_client.get_all_documents("users")
    assert len(users) == 1
