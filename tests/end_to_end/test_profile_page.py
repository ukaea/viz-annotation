import pytest
import requests

pytest.importorskip("playwright")
from playwright.sync_api import expect

from tests.endpoints import create_user
from tests.end_to_end.conftest import login_as


def test_profile_shows_own_username_and_role_read_only(
    server_setup, admin_token, browser
):
    create_user("profuser1", "profuser1_pass")
    user_page = login_as(browser, "profuser1", "profuser1_pass")
    user_page.goto("http://localhost:8002/ui/profile")

    # Username/Role are rendered as one combined, read-only paragraph.
    summary = user_page.locator("p").filter(has_text="Username:")
    expect(summary).to_be_visible()
    expect(summary).to_contain_text("profuser1")
    expect(summary).to_contain_text("Role:")
    expect(summary).to_contain_text("user")
    # No form control for username/role — only Email and password fields exist.
    expect(user_page.get_by_role("textbox", name="Username")).to_be_hidden()

    user_page.context.close()


def test_user_can_update_own_email(server_setup, admin_token, browser):
    create_user("profuser2", "profuser2_pass")
    user_page = login_as(browser, "profuser2", "profuser2_pass")
    user_page.goto("http://localhost:8002/ui/profile")

    user_page.get_by_role("textbox", name="Email").fill("profuser2@example.com")
    user_page.get_by_role("button", name="Save Email").click()
    expect(user_page.get_by_text("Email updated")).to_be_visible()

    user_page.reload()
    expect(user_page.get_by_role("textbox", name="Email")).to_have_value(
        "profuser2@example.com"
    )
    user_page.context.close()


def test_password_mismatch_is_rejected(server_setup, admin_token, browser):
    create_user("profuser3", "profuser3_pass")
    user_page = login_as(browser, "profuser3", "profuser3_pass")
    user_page.goto("http://localhost:8002/ui/profile")

    user_page.get_by_role("textbox", name="New password", exact=True).fill(
        "newpassword123"
    )
    user_page.get_by_role("textbox", name="Confirm new password").fill("different123")
    user_page.get_by_role("button", name="Change Password").click()
    expect(user_page.get_by_text("Passwords do not match")).to_be_visible()

    user_page.context.close()


def test_password_too_short_is_rejected(server_setup, admin_token, browser):
    create_user("profuser4", "profuser4_pass")
    user_page = login_as(browser, "profuser4", "profuser4_pass")
    user_page.goto("http://localhost:8002/ui/profile")

    user_page.get_by_role("textbox", name="New password", exact=True).fill("short1")
    user_page.get_by_role("textbox", name="Confirm new password").fill("short1")
    user_page.get_by_role("button", name="Change Password").click()
    expect(
        user_page.get_by_text("Password must be at least 8 characters")
    ).to_be_visible()

    user_page.context.close()


def test_user_can_change_own_password(server_setup, admin_token, browser):
    create_user("profuser5", "profuser5_pass")
    user_page = login_as(browser, "profuser5", "profuser5_pass")
    user_page.goto("http://localhost:8002/ui/profile")

    user_page.get_by_role("textbox", name="New password", exact=True).fill(
        "newpassword123"
    )
    user_page.get_by_role("textbox", name="Confirm new password").fill("newpassword123")
    user_page.get_by_role("button", name="Change Password").click()
    expect(user_page.get_by_text("Password changed")).to_be_visible()
    user_page.context.close()

    resp = requests.post(
        "http://localhost:8002/auth/token",
        data={"username": "profuser5", "password": "newpassword123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200

    # The old password no longer works.
    resp = requests.post(
        "http://localhost:8002/auth/token",
        data={"username": "profuser5", "password": "profuser5_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 401
