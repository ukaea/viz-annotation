import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect

from tests.endpoints import create_user


def test_login_success(server_setup, guest_page):
    guest_page.goto("http://localhost:8002/ui/login")
    guest_page.get_by_role("textbox", name="Username").fill("admin")
    guest_page.get_by_role("textbox", name="Password").fill("admin")
    guest_page.get_by_role("button", name="Sign In").click()
    expect(guest_page).to_have_url("http://localhost:8002/ui/projects/", timeout=3000)


def test_login_wrong_password(server_setup, guest_page):
    guest_page.goto("http://localhost:8002/ui/login")
    guest_page.get_by_role("textbox", name="Username").fill("admin")
    guest_page.get_by_role("textbox", name="Password").fill("wrong_password")
    guest_page.get_by_role("button", name="Sign In").click()
    expect(guest_page.get_by_text("Invalid username or password")).to_be_visible()
    expect(guest_page).to_have_url("http://localhost:8002/ui/login")


def test_login_unknown_user(server_setup, guest_page):
    guest_page.goto("http://localhost:8002/ui/login")
    guest_page.get_by_role("textbox", name="Username").fill("ghost")
    guest_page.get_by_role("textbox", name="Password").fill("doesnt_matter")
    guest_page.get_by_role("button", name="Sign In").click()
    expect(guest_page.get_by_text("Invalid username or password")).to_be_visible()
    expect(guest_page).to_have_url("http://localhost:8002/ui/login")


def test_login_regular_user_succeeds(server_setup, admin_token, guest_page):
    create_user("dave", "dave_pass123")
    guest_page.goto("http://localhost:8002/ui/login")
    guest_page.get_by_role("textbox", name="Username").fill("dave")
    guest_page.get_by_role("textbox", name="Password").fill("dave_pass123")
    guest_page.get_by_role("button", name="Sign In").click()
    expect(guest_page).to_have_url("http://localhost:8002/ui/projects/", timeout=3000)


def test_login_button_disabled_with_empty_fields(server_setup, guest_page):
    guest_page.goto("http://localhost:8002/ui/login")
    expect(guest_page.get_by_role("button", name="Sign In")).to_be_disabled()
    guest_page.get_by_role("textbox", name="Username").fill("admin")
    expect(guest_page.get_by_role("button", name="Sign In")).to_be_disabled()
    guest_page.get_by_role("textbox", name="Password").fill("admin")
    expect(guest_page.get_by_role("button", name="Sign In")).to_be_enabled()


def test_already_logged_in_user_redirected_away_from_login(server_setup, page):
    # `page` is pre-authenticated as admin (see tests/end_to_end/conftest.py).
    page.goto("http://localhost:8002/ui/login")
    expect(page).to_have_url("http://localhost:8002/ui/projects/", timeout=3000)


def test_logged_out_direct_nav_to_protected_page_redirects_to_login(
    server_setup, guest_page
):
    guest_page.goto("http://localhost:8002/ui/projects/")
    expect(guest_page).to_have_url("http://localhost:8002/ui/login", timeout=3000)


def test_logged_out_direct_nav_to_project_url_redirects_to_login(
    server_setup, guest_page
):
    # A made-up project id — RequireAuth should redirect before this is ever
    # fetched, regardless of whether the id is real.
    guest_page.goto("http://localhost:8002/ui/projects/000000000000000000000000")
    expect(guest_page).to_have_url("http://localhost:8002/ui/login", timeout=3000)


def test_logged_out_direct_nav_to_admin_page_redirects_to_login(
    server_setup, guest_page
):
    guest_page.goto("http://localhost:8002/ui/admin/users")
    expect(guest_page).to_have_url("http://localhost:8002/ui/login", timeout=3000)
