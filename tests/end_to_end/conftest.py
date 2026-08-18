import pytest
import requests

try:
    from playwright.sync_api import Page
except ImportError:
    Page = None  # type: ignore[assignment,misc]

TOKEN_KEY = "tt_access_token"


def login_as(browser, username: str, password: str) -> Page:
    """Return a fresh page authenticated as a specific (non-default) user.

    Logs in via the real API (not the UI form) and seeds the token into a
    brand-new browser context via add_init_script, so it's present before
    that context's very first navigation. Deliberately a new context rather
    than reusing/reloading `page` — `page`'s own add_init_script (below)
    re-fires on every navigation and would just overwrite a token set via
    page.evaluate() + reload() on the next load.
    """
    response = requests.post(
        "http://localhost:8002/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    context = browser.new_context()
    user_page = context.new_page()
    user_page.add_init_script(f"window.localStorage.setItem({TOKEN_KEY!r}, {token!r});")
    return user_page


@pytest.fixture
def page(page: Page, admin_token: str) -> Page:
    """pytest-playwright's page, pre-authenticated as the bootstrap admin.

    Every existing/new e2e test that just needs *some* logged-in session
    gets one for free via this override — no per-test login boilerplate.
    Tests that need a specific non-admin identity should call login_as();
    tests that need a genuinely logged-out browser should use guest_page.
    """
    page.add_init_script(
        f"window.localStorage.setItem({TOKEN_KEY!r}, {admin_token!r});"
    )
    return page


@pytest.fixture
def guest_page(browser) -> Page:
    """A fresh, unauthenticated page/context — for testing the login flow
    itself and other logged-out scenarios. Deliberately bypasses the `page`
    override above, which always seeds a valid admin token.
    """
    context = browser.new_context()
    guest_page = context.new_page()
    yield guest_page
    context.close()
