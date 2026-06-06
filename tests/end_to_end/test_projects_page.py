import pytest
pytest.importorskip("playwright")
import requests
from playwright.sync_api import Page, expect
import re
from datetime import datetime
import time
from tests.endpoints import create_project
import pytest


def check_base_page(page):
    # Expect page is called TokTagger
    expect(page).to_have_title("TokTagger")

    # Expect breadcrumbs at the top to be visible, and showing Projects link
    expect(page.get_by_role("link", name="Projects")).to_be_visible()

    # Expect table to have heading 'Projects'
    expect(page.get_by_role("heading", name="Projects")).to_be_visible()

    # Expect all table headings visible
    expect(page.get_by_role("columnheader", name="Name")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Task")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Date Created")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Loader")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Actions")).to_be_visible()

    # Expect Create, Edit, Delete buttons visible
    expect(page.get_by_role("button", name="Create")).to_be_visible()

    # Expect searchbar visible
    expect(page.get_by_role("searchbox", name="Search By Name")).to_be_visible()

    # Expect page navigation to be visible
    expect(page.get_by_role("button", name="Previous")).to_be_visible()
    expect(page.get_by_role("button", name="Next")).to_be_visible()
    expect(page.get_by_role("button", name="Projects per Page:")).to_be_visible()

    # Expect to be on page 1
    expect(page.get_by_text("Page: 1")).to_be_visible()


def test_empty_projects_table(server_setup, page: Page):
    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expect Edit and Delete buttons NOT visible since no projects in table
    expect(page.get_by_role("button", name="Edit")).to_be_hidden()
    expect(page.get_by_role("button", name="Delete")).to_be_hidden()

    # Expect next and previous buttons to be disabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next")).to_be_disabled()


def test_single_project(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "uda")

    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expect Edit and Delete buttons visible
    expect(page.get_by_role("button", name="Edit")).to_be_visible()
    expect(page.get_by_role("button", name="Delete")).to_be_visible()

    # Expect next and previous buttons to be disabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next")).to_be_disabled()

    # Check project information is shown
    expect(page.get_by_text("Test Project")).to_be_visible()
    expect(page.get_by_text("time-series")).to_be_visible()
    expect(page.get_by_text("uda")).to_be_visible()
    expect(page.get_by_text(re.compile(f"^{datetime.now().date()}*"))).to_be_visible()

    # Expect that I can click on the row in the table and it takes me to a sample page
    table_row = page.get_by_role("rowheader", name="Test Project")
    expect(table_row).to_be_visible()
    expect(table_row).to_be_enabled()

    # Try clicking the row
    table_row.click()
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}", timeout=3000
    )


def test_projects_page_navigation(server_setup, page: Page):
    # Create 6 projects
    for i in range(1, 7):
        create_project(f"Test Project {i}", "time-series", "uda")

    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expect next and previous buttons to be disabled since by default shows 10 projects
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next")).to_be_disabled()

    # Check all projects available
    expect(page.get_by_text("Test Project 1")).to_be_visible()
    expect(page.get_by_text("Test Project 6")).to_be_visible()

    # Try clicking Projects per Page dropdown
    page.get_by_role("button", name="Projects per Page:").click()
    page.get_by_role("option", name="5", exact=True).click()

    # By default the projects appear newest first, so first project shouldn't be on list
    expect(page.get_by_text("Test Project 6")).to_be_visible()
    expect(page.get_by_text("Test Project 1")).to_be_hidden()

    # Previous button should still be disabled, next button should be enabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next")).to_be_enabled()

    # Try pressing Next button
    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("Page: 2")).to_be_visible()

    # Check project 1 is visible, project 2 is not
    expect(page.get_by_text("Test Project 1")).to_be_visible()
    expect(page.get_by_text("Test Project 2")).to_be_hidden()

    # Check Previous button enabled, Next button is not
    expect(page.get_by_role("button", name="Previous")).to_be_enabled()
    expect(page.get_by_role("button", name="Next")).to_be_disabled()

    # Press previous, check we go back to before
    page.get_by_role("button", name="Previous").click()
    expect(page.get_by_text("Page: 1")).to_be_visible()

    expect(page.get_by_text("Test Project 6")).to_be_visible()
    expect(page.get_by_text("Test Project 1")).to_be_hidden()

    # Press Next
    page.get_by_role("button", name="Next").click()
    expect(page.get_by_text("Page: 2")).to_be_visible()

    # Now change projects per page back to 10, check we are sent back to page 1
    page.get_by_role("button", name="Projects per Page:").click()
    page.get_by_role("option", name="10", exact=True).click()

    # Should be back to page 1 with all projects visible
    expect(page.get_by_text("Page: 1")).to_be_visible()
    expect(page.get_by_text("Test Project 1")).to_be_visible()
    expect(page.get_by_text("Test Project 6")).to_be_visible()

    # And both buttons disabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next")).to_be_disabled()


def test_projects_sorting(server_setup, page: Page):
    def sort(page, col_name, expected_first, expected_second):
        # Sort by given column
        page.get_by_role("columnheader", name=col_name).click()

        # Check projects in correct order
        expect(page.get_by_role("row").nth(1)).to_contain_text(expected_first)
        expect(page.get_by_role("row").nth(2)).to_contain_text(expected_second)

        # Click again, sort in opposite direction
        page.get_by_role("columnheader", name=col_name).click()

        # PCheck projects in reverse direction
        expect(page.get_by_role("row").nth(1)).to_contain_text(expected_second)
        expect(page.get_by_role("row").nth(2)).to_contain_text(expected_first)

    # Create some projects
    create_project("A Project", "video", "uda")
    time.sleep(0.1)
    create_project("B Project", "time-series", "tabular")

    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Sort by Name: A should be first, then B
    sort(page, "Name", "A Project", "B Project")

    # Sort by Task, B should be first (time-series), then A (video)
    sort(page, "Task", "B Project", "A Project")

    # Sort by Timestamp, A should be first (oldest - so lowest timestamp), then B (newest - highest timestamp)
    sort(page, "Date Created", "A Project", "B Project")

    # Sort by Loader, B should be first (parquet), then A (uda)
    sort(page, "Loader", "B Project", "A Project")


def test_projects_search(server_setup, page: Page):
    # Create some projects with different names
    create_project("Project A", "time-series", "uda")
    create_project("Project B", "time-series", "uda")
    create_project("project C", "time-series", "uda")
    create_project("Test Project", "time-series", "uda")
    create_project("Projection", "time-series", "uda")
    create_project("New UDA ELMs", "time-series", "uda")

    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    searchbox = page.get_by_role("searchbox", name="Search By Name")

    # Search for Project 1
    searchbox.fill("Project A")
    searchbox.press("Enter")

    # Check there is only one row
    expect(page.get_by_role("row").nth(1)).to_contain_text("Project A")
    expect(page.get_by_role("row").nth(2)).to_be_hidden()

    # Search for project
    searchbox.fill("project")
    searchbox.press("Enter")

    # Search should be case insensitive, find any entries which contain that phrase
    # So 5 projects, not 'New UDA Elms'
    # In newest first order
    expect(page.get_by_role("row").nth(1)).to_contain_text("Projection")
    expect(page.get_by_role("row").nth(2)).to_contain_text("Test Project")
    expect(page.get_by_role("row").nth(3)).to_contain_text("project C")
    expect(page.get_by_role("row").nth(4)).to_contain_text("Project B")
    expect(page.get_by_role("row").nth(5)).to_contain_text("Project A")
    expect(page.get_by_role("row").nth(6)).to_be_hidden()

    # Search for 'Project ', with a space
    searchbox.fill("Project ")
    searchbox.press("Enter")

    # Should find Project A, Project B, Project C
    expect(page.get_by_role("row").nth(1)).to_contain_text("project C")
    expect(page.get_by_role("row").nth(2)).to_contain_text("Project B")
    expect(page.get_by_role("row").nth(3)).to_contain_text("Project A")
    expect(page.get_by_role("row").nth(4)).to_be_hidden()

    # Search for 'wrong'
    searchbox.fill("wrong")
    searchbox.press("Enter")

    # Should find nothing
    expect(page.get_by_role("row").nth(1)).to_be_hidden()

    # Enter blank string (delete search)
    searchbox.fill("")
    searchbox.press("Enter")

    # Should find all 6 entries
    expect(page.get_by_role("row").nth(6)).to_contain_text("Project A")
    expect(page.get_by_role("row").nth(7)).to_be_hidden()


def test_delete_project(server_setup, page: Page):
    # Create some projects
    create_project("Project A", "time-series", "uda")
    create_project("Project B", "time-series", "tabular")

    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press delete next to project B
    page.get_by_role("row").nth(1).get_by_role("button", name="Delete").click()

    # Check warning modal opens
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(modal.get_by_role("heading", name="Confirm Deletion")).to_be_visible()

    expect(
        modal.get_by_text("Are you sure you want to delete project Project B?")
    ).to_be_visible()

    expect(modal.get_by_role("button", name="Cancel")).to_be_visible()
    expect(modal.get_by_role("button", name="Delete")).to_be_visible()

    # Press cancel, check samples still exist
    modal.get_by_role("button", name="Cancel").click()

    # Check both projects still present
    expect(page.get_by_role("row").nth(1)).to_contain_text("Project B")
    expect(page.get_by_role("row").nth(2)).to_contain_text("Project A")

    # Press delete next to project B
    page.get_by_role("row").nth(1).get_by_role("button", name="Delete").click()

    # Check warning modal opens, click delete
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    modal.get_by_role("button", name="Delete").click()

    # Check row 1 is now Project A, and it is the only row
    expect(page.get_by_role("row").nth(1)).to_contain_text("Project A")
    expect(page.get_by_role("row").nth(2)).to_be_hidden()

    # Now delete project A
    page.get_by_role("row").nth(1).get_by_role("button", name="Delete").click()

    # Check warning modal opens, click delete
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    modal.get_by_role("button", name="Delete").click()

    # Check no projects remain
    expect(page.get_by_role("row").nth(1)).to_be_hidden()


def test_edit_project(server_setup, page: Page):
    # Create some projects
    create_project("Test Project", "time-series", "uda")

    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press edit button
    page.get_by_role("row").nth(1).get_by_role("button", name="Edit").click()

    # Modal should have opened
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()

    # Check edit project modal has opened
    expect(modal.get_by_role("heading", name="Edit Project")).to_be_visible()
    expect(modal.get_by_role("button", name="Save Changes")).to_be_visible()
    expect(modal.get_by_role("button", name="Cancel")).to_be_visible()

    # Check you can edit project name and query strategy
    expect(modal.get_by_role("textbox", name="Project Name")).to_be_visible()
    expect(modal.get_by_text("Query Strategy")).to_be_visible()
    expect(modal.get_by_role("radio", name="Sequential")).to_be_enabled()

    # Check you cannot edit project task or data loader
    expect(modal.get_by_role("combobox", name="Task")).to_be_disabled()
    expect(modal.get_by_role("combobox", name="Data Loader")).to_be_disabled()

    # Open annotation label settings, check they are editable
    modal.get_by_role("button", name="Annotation Label Settings").click()
    expect(modal.get_by_role("textbox", name="Shot Labels")).to_be_editable()
    expect(modal.get_by_role("textbox", name="Time Region Labels")).to_be_editable()
    expect(modal.get_by_role("textbox", name="Time Point Labels")).to_be_editable()

    # Open Time Range settings, check they are editable
    modal.get_by_role("button", name="Time Range Settings").click()
    expect(modal.get_by_role("textbox", name="Time Min (s)")).to_be_editable()
    expect(modal.get_by_role("textbox", name="Time Max (s)")).to_be_editable()
    expect(modal.get_by_role("textbox", name="Min Time Step (s)")).to_be_editable()

    # Edit the project name
    modal.get_by_role("textbox", name="Project Name").fill("Updated Project")

    # Change the query strategy
    modal.get_by_role("radio", name="Sequential").click()

    # Save changes
    modal.get_by_role("button", name="Save Changes").click()

    # Check project name updated
    expect(page.get_by_role("row").nth(1)).to_contain_text("Updated Project")

    # Get back project from server to check it updated
    response = requests.get("http://localhost:8002/projects")
    project = response.json()[0]
    assert project["name"] == "Updated Project"
    assert project["query_strategy"] == "sequential"
    assert project["task"] == "time-series"
    assert project["data_loader"] == "uda"


@pytest.mark.parametrize(
    ("task", "data_loader", "time_range_visible", "expected_annotation_label_types"),
    [
        (
            "time-series",
            "tabular",
            True,
            ["Shot Labels", "Time Region Labels", "Time Point Labels"],
        ),
        (
            "time-series",
            "uda",
            True,
            ["Shot Labels", "Time Region Labels", "Time Point Labels"],
        ),
        # (
        #     "spectrogram",
        #     "sal",
        #     True,
        #     [
        #         "Shot Labels",
        #         "Time Region Labels",
        #         "Time Point Labels",
        #         "Bounding Box Labels",
        #         "Polygon Labels",
        #     ],
        # ),
        # (
        #     "spectrogram",
        #     "fair_mast",
        #     True,
        #     [
        #         "Shot Labels",
        #         "Time Region Labels",
        #         "Time Point Labels",
        #         "Bounding Box Labels",
        #         "Polygon Labels",
        #     ],
        # ),
        (
            "video",
            "image",
            False,
            [
                "Shot Labels",
                "Video Bounding Box Labels",
            ],
        ),
        (
            "video",
            "uda_camera",
            False,
            [
                "Shot Labels",
                "Video Bounding Box Labels",
            ],
        ),
    ],
)
def test_create_project(
    server_setup,
    page: Page,
    task: str,
    data_loader: str,
    time_range_visible: bool,
    expected_annotation_label_types: list[str],
):
    non_expected_annotation_label_types = [
        item
        for item in [
            "Shot Labels",
            "Time Region Labels",
            "Time Point Labels",
            "Bounding Box Labels",
            "Polygon Labels",
            "Video Bounding Box Labels",
        ]
        if item not in expected_annotation_label_types
    ]
    # Navigate to page
    page.goto("http://localhost:8002")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press create button
    page.get_by_role("button", name="Create").click()

    # Check modal has opened
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()

    # Check it appears as expected
    expect(modal.get_by_role("heading", name="Create Project")).to_be_visible()
    expect(modal.get_by_role("button", name="Create")).to_be_visible()
    expect(modal.get_by_role("button", name="Cancel")).to_be_visible()

    # Check form contains required fields
    expect(modal.get_by_role("textbox", name="Project Name")).to_be_visible()
    expect(modal.get_by_role("combobox", name="Data Loader")).to_be_visible()
    expect(modal.get_by_role("combobox", name="Task")).to_be_visible()

    expect(modal.get_by_role("radio", name="Sequential")).to_be_visible()
    expect(modal.get_by_role("radio", name="Random")).to_be_visible()
    expect(modal.get_by_role("radio", name="Uncertainty Sampling")).to_be_visible()

    # Fill in common info
    modal.get_by_role("button", name="Task").click()
    page.get_by_role("option", name=task).click()

    modal.get_by_role("button", name="Data Loader").click()
    page.get_by_role("option", name=data_loader, exact=True).click()

    modal.get_by_role("radio", name="Random").click()
    modal.get_by_role("textbox", name="Project Name").fill("Test Project")

    # Check if Time Range settings appear
    if time_range_visible:
        expect(modal.get_by_role("button", name="Time Range Settings")).to_be_visible()
        modal.get_by_role("button", name="Time Range Settings").click()

        modal.get_by_role("textbox", name="Time Min (s)").fill("1")
        modal.get_by_role("textbox", name="Time Max (s)").fill("5")
        modal.get_by_role("textbox", name="Min Time Step (s)").fill("0.1")

        modal.get_by_role("button", name="Time Range Settings").click()
    else:
        expect(modal.get_by_role("button", name="Time Range Settings")).to_be_hidden()

    # Check correct annotation label boxes present
    modal.get_by_role("button", name="Annotation Label Settings").click()

    for label_type in expected_annotation_label_types:
        expect(
            modal.get_by_role("textbox", name=label_type, exact=True)
        ).to_be_visible()
        modal.get_by_role("textbox", name=label_type).fill(label_type)

    for label_type in non_expected_annotation_label_types:
        expect(modal.get_by_role("textbox", name=label_type, exact=True)).to_be_hidden()

    # Press create
    modal.get_by_role("button", name="Create").click()

    # Check project added to table
    expect(page.get_by_role("row").nth(1)).to_contain_text("Test Project")

    # Get project from API, check details are all correct
    response = requests.get("http://localhost:8002/projects")
    project = response.json()[0]
    assert project["name"] == "Test Project"
    assert project["query_strategy"] == "random"
    assert project["task"] == task
    assert project["data_loader"] == data_loader
    if time_range_visible:
        assert project["time_min"] == 1
        assert project["time_max"] == 5
        assert project["min_time_step"] == 0.1

    for label_type in expected_annotation_label_types:
        assert project[f"{label_type.lower().replace(' ', '_')}"] == [label_type]

    for label_type in non_expected_annotation_label_types:
        # These are assigned default values by the server
        assert project[f"{label_type.lower().replace(' ', '_')}"] != [label_type]
