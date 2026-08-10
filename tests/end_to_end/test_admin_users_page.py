import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect

from tests.end_to_end.conftest import login_as
from tests.endpoints import create_user, get_user


def _user_row(page, username):
    # Matching on the row's own text (has_text=username) is ambiguous once
    # another row's *role* column also happens to read e.g. "admin" — scope
    # to the rowheader (username/first column) specifically instead.
    return page.get_by_role("row").filter(
        has=page.get_by_role("rowheader", name=username, exact=True)
    )


def test_admin_can_create_user(server_setup, page):
    page.goto("http://localhost:8002/ui/admin/users")
    expect(page.get_by_role("heading", name="User Management")).to_be_visible()

    page.get_by_role("button", name="Add User").click()
    page.get_by_role("textbox", name="Username").fill("newuser")
    page.get_by_role("textbox", name="Password").fill("newpass123")
    page.get_by_role("button", name="Create").click()

    row = _user_row(page, "newuser")
    expect(row).to_be_visible()


def test_admin_can_change_user_role(server_setup, admin_token, page):
    create_user("promoteme", "pass123")
    page.goto("http://localhost:8002/ui/admin/users")

    row = _user_row(page, "promoteme")
    expect(row.get_by_text("user")).to_be_visible()

    row.get_by_role("button", name="Edit").click()
    page.get_by_role("button", name="Global Role").click()
    page.get_by_role("option", name="Admin", exact=True).click()
    page.get_by_role("button", name="Save").click()

    expect(row.get_by_text("admin")).to_be_visible()


def test_admin_can_deactivate_and_reactivate_user(server_setup, admin_token, page):
    user_id = create_user("flipflop", "pass123")
    page.goto("http://localhost:8002/ui/admin/users")

    row = _user_row(page, "flipflop")
    row.get_by_role("button", name="Deactivate").click()
    expect(row.get_by_role("button", name="Activate")).to_be_visible()
    assert get_user(user_id)["is_active"] is False

    row.get_by_role("button", name="Activate").click()
    expect(row.get_by_role("button", name="Deactivate")).to_be_visible()
    assert get_user(user_id)["is_active"] is True


def test_admin_can_delete_user(server_setup, admin_token, page):
    create_user("deleteme", "pass123")
    page.goto("http://localhost:8002/ui/admin/users")

    row = _user_row(page, "deleteme")
    row.get_by_role("button", name="Delete").click()
    # Every row's own Delete button is still technically present (and
    # "visible") behind the confirmation dialog, so scope to the dialog.
    page.get_by_role("dialog").get_by_role("button", name="Delete", exact=True).click()

    expect(row).to_be_hidden()


def test_admin_cannot_deactivate_or_delete_self(server_setup, page):
    page.goto("http://localhost:8002/ui/admin/users")

    row = _user_row(page, "admin")
    expect(row.get_by_role("button", name="Deactivate")).to_be_disabled()
    expect(row.get_by_role("button", name="Delete")).to_be_disabled()


def test_non_admin_cannot_access_admin_users_page(server_setup, admin_token, browser):
    create_user("regular", "regular_pass")
    regular_page = login_as(browser, "regular", "regular_pass")
    regular_page.goto("http://localhost:8002/ui/admin/users")
    expect(regular_page).to_have_url("http://localhost:8002/ui/projects/", timeout=3000)
    regular_page.context.close()
