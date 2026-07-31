import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect

from tests.endpoints import (
    add_project_member,
    create_project,
    create_user,
    get_project_members,
)
from tests.end_to_end.conftest import login_as


def _open_members_dialog(page, project_id):
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")
    page.get_by_role("button", name="Manage Members").click()


def _member_row(page, username):
    # has_text=username alone is ambiguous: react-spectrum Pickers leave a
    # combined-options accessible name ("AdminAnnotatorViewer") behind in the
    # DOM, and that string contains "Admin"/"Viewer" as substrings. Scope to
    # the rowheader (username column) specifically instead.
    return page.get_by_role("row").filter(
        has=page.get_by_role("rowheader", name=username, exact=True)
    )


def test_creator_is_auto_added_as_project_admin_member(server_setup, page):
    project_id = create_project("Auto Member Project", "time-series", "tabular")
    _open_members_dialog(page, project_id)

    row = _member_row(page, "admin")
    expect(row).to_be_visible()
    expect(row.get_by_role("button", name="Admin Role")).to_be_visible()


def test_admin_can_add_member(server_setup, admin_token, page):
    create_user("member_alice", "alice_pass123")
    project_id = create_project("Add Member Project", "time-series", "tabular")
    _open_members_dialog(page, project_id)

    page.get_by_role("textbox", name="Username").fill("member_alice")
    page.get_by_role("button", name="Annotator Role").click()
    page.get_by_role("option", name="Viewer").click()
    page.get_by_role("button", name="Add", exact=True).click()

    row = _member_row(page, "member_alice")
    expect(row).to_be_visible()
    expect(row.get_by_role("button", name="Viewer Role")).to_be_visible()


def test_admin_can_change_member_role(server_setup, admin_token, page):
    create_user("member_bob", "bob_pass123")
    project_id = create_project("Change Role Project", "time-series", "tabular")
    _open_members_dialog(page, project_id)

    page.get_by_role("textbox", name="Username").fill("member_bob")
    page.get_by_role("button", name="Add", exact=True).click()

    row = _member_row(page, "member_bob")
    row.get_by_role("button", name="Annotator Role").click()
    page.get_by_role("option", name="Viewer").click()

    expect(row.get_by_role("button", name="Viewer Role")).to_be_visible()

    members = get_project_members(project_id)
    bob_member = next(m for m in members if m["username"] == "member_bob")
    assert bob_member["role"] == "viewer"


def test_admin_can_remove_member(server_setup, admin_token, page):
    create_user("member_carol", "carol_pass123")
    project_id = create_project("Remove Member Project", "time-series", "tabular")
    _open_members_dialog(page, project_id)

    page.get_by_role("textbox", name="Username").fill("member_carol")
    page.get_by_role("button", name="Add", exact=True).click()

    row = _member_row(page, "member_carol")
    expect(row).to_be_visible()
    row.get_by_role("button", name="Remove").click()

    expect(_member_row(page, "member_carol")).to_be_hidden()


def test_viewer_member_sees_read_only_member_list(server_setup, admin_token, browser):
    create_user("member_dave", "dave_pass123")
    project_id = create_project("Read Only Members Project", "time-series", "tabular")
    add_project_member(project_id, "member_dave", role="viewer")

    dave_page = login_as(browser, "member_dave", "dave_pass123")
    dave_page.goto(f"http://localhost:8002/ui/projects/{project_id}")
    dave_page.get_by_role("button", name="Manage Members").click()

    # Viewer cannot manage members: no Role picker or Remove button, just text.
    expect(dave_page.get_by_role("button", name="Add", exact=True)).to_be_hidden()
    expect(dave_page.get_by_role("button", name="Remove")).to_be_hidden()
    row = _member_row(dave_page, "member_dave")
    expect(row.get_by_text("viewer")).to_be_visible()

    dave_page.context.close()
