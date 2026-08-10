"""
Direct-navigation access control: what happens when you paste a project/sample
URL into the address bar without the credentials/membership to see it.

  - Not logged in at all -> redirected to /ui/login (RequireAuth in App.jsx).
  - Logged in, but not a member of that project -> a clear "Error" panel
    ("You are not a member of this project"), not a blank or broken page.
  - A project id that doesn't exist at all -> the same clean Error panel,
    with the backend's "not found" message.
"""

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect

from tests.end_to_end.conftest import login_as
from tests.endpoints import (
    add_project_member,
    create_local_samples,
    create_project,
    create_user,
)


def test_logged_out_project_url_redirects_to_login(
    server_setup, admin_token, guest_page
):
    project_id = create_project("Logged Out Project", "time-series", "tabular")
    guest_page.goto(f"http://localhost:8002/ui/projects/{project_id}")
    expect(guest_page).to_have_url("http://localhost:8002/ui/login", timeout=3000)


def test_logged_out_sample_url_redirects_to_login(
    server_setup, admin_token, guest_page
):
    project_id = create_project("Logged Out Sample Project", "time-series", "tabular")
    sample_ids = create_local_samples(project_id, [1], ".", ["Ip"], ["10000.parquet"])
    guest_page.goto(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}"
    )
    expect(guest_page).to_have_url("http://localhost:8002/ui/login", timeout=3000)


def test_non_member_sees_access_denied_not_blank_page(
    server_setup, admin_token, browser
):
    create_user("outsider1", "outsider1_pass")
    project_id = create_project("Members Only Project", "time-series", "tabular")

    outsider_page = login_as(browser, "outsider1", "outsider1_pass")
    outsider_page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Still on the project URL (RequireAuth passed — outsider1 IS logged in),
    # but the page shows a clear error instead of blank/garbled content.
    expect(outsider_page).to_have_url(f"http://localhost:8002/ui/projects/{project_id}")
    expect(outsider_page.get_by_text("Error")).to_be_visible()
    expect(
        outsider_page.get_by_text("You are not a member of this project")
    ).to_be_visible()
    outsider_page.context.close()


def test_non_member_sees_access_denied_on_sample_view(
    server_setup, admin_token, browser
):
    create_user("outsider2", "outsider2_pass")
    project_id = create_project("Members Only Sample Project", "time-series", "tabular")
    sample_ids = create_local_samples(project_id, [1], ".", ["Ip"], ["10000.parquet"])

    outsider_page = login_as(browser, "outsider2", "outsider2_pass")
    outsider_page.goto(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}"
    )

    expect(outsider_page.get_by_text("Error")).to_be_visible()
    expect(
        outsider_page.get_by_text("You are not a member of this project")
    ).to_be_visible()
    outsider_page.context.close()


def test_nonexistent_project_shows_not_found_not_blank_page(server_setup, page):
    page.goto("http://localhost:8002/ui/projects/000000000000000000000000")
    expect(page.get_by_text("Error")).to_be_visible()
    expect(page.get_by_text("Project not found with that ID.")).to_be_visible()


def test_viewer_member_can_access_project(server_setup, admin_token, browser):
    """Sanity check the negative-path tests above against a real positive case
    — a genuine member should see the actual project, not an error."""
    create_user("insider1", "insider1_pass")
    project_id = create_project("Accessible Project", "time-series", "tabular")
    add_project_member(project_id, "insider1", role="viewer")

    insider_page = login_as(browser, "insider1", "insider1_pass")
    insider_page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    expect(insider_page.get_by_text("Error")).to_be_hidden()
    expect(insider_page.get_by_role("heading", name="Samples")).to_be_visible()
    insider_page.context.close()
