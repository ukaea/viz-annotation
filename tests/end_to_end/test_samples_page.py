from playwright.sync_api import Page, expect
import re
from datetime import datetime
import pathlib
from tests.endpoints import (
    create_project,
    create_local_samples,
    create_uda_samples,
    create_model_samples,
    create_query_strategy_samples,
)
from tests.end_to_end import form_check
import time
import requests
import tempfile
import pytest
import json
import csv
from toktagger.api.schemas.annotations import TimePoint, TimeRegion
import toktagger.api.config as config


def check_base_page(page):
    # Expect page is called TokTagger
    expect(page).to_have_title("TokTagger")

    # Expect breadcrumbs at the top to be visible, and showing Projects link
    expect(page.get_by_role("link", name="Projects")).to_be_visible()

    # Expect table to have heading 'Samples'
    expect(page.get_by_role("heading", name="Samples")).to_be_visible()

    # Expect all table headings visible
    expect(page.get_by_role("columnheader", name="Shot ID")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Date Created")).to_be_visible()

    # Expect Add and Clear Samples button visible
    expect(page.get_by_role("button", name="Add Samples", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Clear Samples", exact=True)).to_be_visible()

    # Expect Model Train/Load/Predict buttons visible
    expect(
        page.get_by_role("button", name="Create Predictions from ML Model")
    ).to_be_visible()
    expect(page.get_by_role("button", name="Train ML Model")).to_be_visible()
    expect(page.get_by_role("button", name="Load ML Model")).to_be_visible()
    expect(page.get_by_role("button", name="ML Model Help")).to_be_visible()

    # Expect searchbar visible
    expect(page.get_by_role("searchbox", name="Search By Shot ID")).to_be_visible()

    # Expect page navigation to be visible
    expect(page.get_by_role("button", name="Previous")).to_be_visible()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Samples per Page:")).to_be_visible()

    # Expect to be on page 1
    expect(page.get_by_text("Page: 1")).to_be_visible()

    # Expect import and export annotation buttons to be visible
    expect(page.get_by_role("button", name="Import Annotations")).to_be_visible()
    expect(page.get_by_role("button", name="Export Annotations")).to_be_visible()

    # Expect Jump to Next Shot button to be visible
    expect(page.get_by_role("button", name="Jump to Next Sample")).to_be_visible()


def test_empty_samples_table(server_setup, page: Page):
    # Create Project
    project_id = create_project("Test Project", "time-series", "tabular")

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expect next and previous buttons to be disabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_disabled()

    # Check we can navigate back to Projects page via breadcrumbs
    page.get_by_role("link", name="Projects").click()
    expect(page).to_have_url("http://localhost:8002/ui/projects", timeout=3000)


def test_single_sample(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And a sample
    ids = create_local_samples(
        project_id, [10000], pathlib.Path(__file__).parents[1], ["Ip"]
    )
    sample_id = ids[0]

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expect next and previous buttons to be disabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_disabled()

    # Check sample information is shown
    expect(page.get_by_text("10000", exact=True)).to_be_visible()
    expect(page.get_by_text(re.compile(f"^{datetime.now().date()}*"))).to_be_visible()

    # Expect that I can click on the row in the table and it takes me to a ELM view page
    table_row = page.get_by_role("rowheader", name="10000")
    expect(table_row).to_be_visible()
    expect(table_row).to_be_enabled()

    # Try clicking the row
    table_row.click()
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}?sortColumn=shot_id&sortDirection=ascending",
        timeout=3000,
    )


def test_delete_sample(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And a sample
    create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press delete next to 10000
    page.get_by_role("row").nth(1).get_by_role("button", name="Delete").click()

    # Check warning modal opens
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(modal.get_by_role("heading", name="Confirm Deletion")).to_be_visible()

    expect(
        modal.get_by_text("Are you sure you want to delete sample with shot ID 10000?")
    ).to_be_visible()

    expect(modal.get_by_role("button", name="Cancel")).to_be_visible()
    expect(modal.get_by_role("button", name="Delete")).to_be_visible()

    # Press cancel, check samples still exist
    modal.get_by_role("button", name="Cancel").click()

    expect(page.get_by_role("row").nth(1)).to_contain_text("10000")
    expect(page.get_by_role("row").nth(2)).to_contain_text("10001")

    # Press delete next to 10000 again
    page.get_by_role("row").nth(1).get_by_role("button", name="Delete").click()

    # Check warning modal opens, press delete
    modal = page.get_by_role("dialog")
    modal.get_by_role("button", name="Delete").click()

    # Check row 1 is now Project A, and it is the only row
    expect(page.get_by_role("row").nth(1)).to_contain_text("10001")
    expect(page.get_by_role("row").nth(2)).to_be_hidden()

    # Now delete 10001
    page.get_by_role("row").nth(1).get_by_role("button", name="Delete").click()

    # Check warning modal opens, press delete
    modal = page.get_by_role("dialog")
    modal.get_by_role("button", name="Delete").click()

    # Check no projects remain
    expect(page.get_by_role("row").nth(1)).to_be_hidden()


def test_sample_page_navigation(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "uda")
    # And 6 samples
    create_uda_samples(project_id, shot_ids=list(range(10001, 10007)))

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expect next and previous buttons to be disabled since by default shows 10 samples
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_disabled()

    # Check all samples available
    expect(page.get_by_text("10001", exact=True)).to_be_visible()
    expect(page.get_by_text("10006", exact=True)).to_be_visible()

    # Try clicking Samples per Page dropdown
    page.get_by_role("button", name="Samples per Page:").click()
    page.get_by_role("option", name="5", exact=True).click()

    # By default the samples appear shot ID low to high, so 10006 shoidnt be on list
    expect(page.get_by_text("10001", exact=True)).to_be_visible()
    expect(page.get_by_text("10006", exact=True)).to_be_hidden()

    # Previous button should still be disabled, next button should be enabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_enabled()

    # Try pressing Next button
    page.get_by_role("button", name="Next", exact=True).click()
    expect(page.get_by_text("Page: 2")).to_be_visible()

    # Check sample 10006 is visible, 10005 is not
    expect(page.get_by_text("10006", exact=True)).to_be_visible()
    expect(page.get_by_text("10005", exact=True)).to_be_hidden()

    # Check Previous button enabled, Next button is not
    expect(page.get_by_role("button", name="Previous")).to_be_enabled()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_disabled()

    # Press previous, check we go back to before
    page.get_by_role("button", name="Previous").click()
    expect(page.get_by_text("Page: 1")).to_be_visible()

    expect(page.get_by_text("10001", exact=True)).to_be_visible()
    expect(page.get_by_text("10006", exact=True)).to_be_hidden()

    # Press Next
    page.get_by_role("button", name="Next", exact=True).click()
    expect(page.get_by_text("Page: 2")).to_be_visible()

    # Now change samples per page back to 10, check we are sent back to page 1
    page.get_by_role("button", name="Samples per Page:").click()
    page.get_by_role("option", name="10", exact=True).click()

    # Should be back to page 1 with all samples visible
    expect(page.get_by_text("Page: 1")).to_be_visible()
    expect(page.get_by_text("10001", exact=True)).to_be_visible()
    expect(page.get_by_text("10006", exact=True)).to_be_visible()

    # And both buttons disabled
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()
    expect(page.get_by_role("button", name="Next", exact=True)).to_be_disabled()


def test_samples_sorting(server_setup, page: Page):
    def sort(page, col_name, expected_first, expected_second):
        # Sort by given column
        page.get_by_role("columnheader", name=col_name).click()

        # Check samples in correct order
        expect(page.get_by_role("row").nth(1)).to_contain_text(expected_first)
        expect(page.get_by_role("row").nth(2)).to_contain_text(expected_second)

        # Click again, sort in opposite direction
        page.get_by_role("columnheader", name=col_name).click()

        # PCheck samples in reverse direction
        expect(page.get_by_role("row").nth(1)).to_contain_text(expected_second)
        expect(page.get_by_role("row").nth(2)).to_contain_text(expected_first)

    # Create a project
    project_id = create_project("Test Project", "time-series", "uda")
    # And 2 samples
    create_uda_samples(project_id, shot_ids=[20000])
    time.sleep(0.1)

    sample_ids = create_uda_samples(project_id, shot_ids=[10000])

    # Set 10000 sample to be validated
    requests.put(
        f"http://localhost:8002/projects/{project_id}/samples",
        json=[{"id": sample_ids[0], "updates": {"validated_annotations": True}}],
    )
    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Sort by Timestamp, 20000 should be first (oldest - so lowest timestamp), then 10000 (newest - highest timestamp)
    sort(page, "Date Created", "20000", "10000")

    # Sort by Shot ID, 10000 should be first, then 20000
    sort(page, "Shot ID", "10000", "20000")

    # Sort by validated annotations - ascending, so should be non validated first
    sort(page, "Validated", "20000", "10000")


def test_samples_search(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "uda")
    # And 6 samples
    create_uda_samples(project_id, shot_ids=list(range(10001, 10007)))

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    searchbox = page.get_by_role("searchbox", name="Search By Shot ID")

    # Search for sample 10001
    searchbox.fill("10001")
    searchbox.press("Enter")

    # Check there is only one row
    expect(page.get_by_role("row").nth(1)).to_contain_text("10001")
    expect(page.get_by_role("row").nth(2)).to_be_hidden()

    # Search for sample 10006
    searchbox.fill("10006")
    searchbox.press("Enter")

    # Check there is only one row
    expect(page.get_by_role("row").nth(1)).to_contain_text("10006")
    expect(page.get_by_role("row").nth(2)).to_be_hidden()

    # Search for '100' - this search is not regex, so this should return nothing
    searchbox.fill("100")
    searchbox.press("Enter")

    # Check there are no results
    expect(page.get_by_role("row").nth(1)).to_be_hidden()

    # Enter blank string (delete search)
    searchbox.fill("")
    searchbox.press("Enter")

    # Should find all 6 entries
    expect(page.get_by_role("row").nth(6)).to_contain_text("10006")
    expect(page.get_by_role("row").nth(7)).to_be_hidden()


@pytest.mark.parametrize(
    ("query_strategy", "expected_next", "sort_by", "sort_direction"),
    [
        (
            "uncertainty",
            4,  # Most uncertain
            "shot_id",
            "ascending",
        ),
        (
            "uncertainty",
            4,  # Most uncertain,
            "shot_id",
            "descending",  # Order of samples shouldn't make any difference
        ),
        (
            "random",
            3,  # Random order of non-validated
            "shot_id",
            "ascending",
        ),
        (
            "random",
            3,  # Random order of non-validated
            "shot_id",
            "descending",
        ),
        (
            "sequential",
            2,  # Increasing order of shot, first non-validated
            "shot_id",
            "ascending",
        ),
        (
            "sequential",
            4,  # Decreasing order of shot, first non validated
            "shot_id",
            "descending",
        ),
        (
            "sequential",
            4,  # Increasing order of time created, first non validated
            "_id",
            "ascending",
        ),
        (
            "sequential",
            2,  # Decreasing order of time created, first non validated
            "_id",
            "descending",
        ),
    ],
)
def test_samples_jump_to_next_button(
    server_setup, page: Page, query_strategy, expected_next, sort_by, sort_direction
):
    project_id = create_project("Test Project", "time-series", "tabular")

    project_id, sample_ids = create_query_strategy_samples(
        query_strategy=query_strategy
    )
    # Go to samples table
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")
    check_base_page(page)

    # Click columns if needed to set ordering
    if sort_by == "_id":
        page.get_by_role("columnheader", name="Date Created").click()
        if sort_direction == "descending":
            page.get_by_role("columnheader", name="Date Created").click()
    elif sort_direction == "descending":
        page.get_by_role("columnheader", name="Shot ID").click()

    # Press Jump to Next Sample
    page.get_by_role("button", name="Jump to Next Sample").click()

    # Check navigated to next sample
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[expected_next]}?sortColumn={sort_by}&sortDirection={sort_direction}"
    )
    expect(page.get_by_label("time-series")).to_be_visible()


def test_create_samples_shot_data_range(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "uda")

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press create button
    page.get_by_role("button", name="Add Samples", exact=True).click()

    # Check modal has opened
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(modal.get_by_role("heading", name="Add Samples to Project")).to_be_visible()

    # Check we can now see Shot Min, Shot Max, and Signal Names
    expect(modal.get_by_text("Shot Min", exact=True)).to_be_visible()
    expect(modal.get_by_text("Shot Max", exact=True)).to_be_visible()
    expect(modal.get_by_role("textbox", name="Signal Names")).to_be_visible()

    # Fill in these fields
    modal.get_by_role("textbox", name="Shot Min").fill("12380")
    modal.get_by_role("textbox", name="Shot Max").fill("12385")

    # Add signal names
    modal.get_by_role("textbox", name="Signal Names").fill("ip, dalpha")

    # Create samples
    modal.get_by_role("button", name="Add Samples").click()

    # Check samples added to table
    expect(page.get_by_role("row").nth(1)).to_contain_text("12380")
    expect(page.get_by_role("row").nth(6)).to_contain_text("12385")

    # Check 6 samples added (12380 to 12385 inclusive)
    response = requests.get(f"http://localhost:8002/projects/{project_id}/samples")
    samples = response.json()
    assert len(samples) == 6
    assert all(
        sorted(sample["data"]["signal_names"]) == ["dalpha", "ip"] for sample in samples
    )


def test_create_samples_shot_data_file(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "uda")

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press create button
    page.get_by_role("button", name="Add Samples", exact=True).click()

    # Check modal has opened
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(modal.get_by_role("heading", name="Add Samples to Project")).to_be_visible()

    # Select Text File option
    modal.get_by_role("radio", name="Text File").click()

    # Check we can now see Choose File button and Signal Names textbox
    expect(modal.get_by_role("button", name="Choose File")).to_be_visible()
    expect(modal.get_by_role("textbox", name="Signal Names")).to_be_visible()

    # Create a Time Point annotation using sample ID in schema
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as file:
        data = [
            ["Shot ID", "Disruption"],
            [12380, True],
            [12381, False],
            [12382, False],
            [12383, True],
            [12384, False],
            [12385, False],
        ]

        # Create a csv.writer object
        writer = csv.writer(file)
        # Write data to the CSV file
        writer.writerows(data)
        file.flush()
        # Make sure file is written
        time.sleep(0.5)

        # Import annotation
        with page.expect_file_chooser() as fc_info:
            page.get_by_role("button", name="Choose File").click()
            file_chooser = fc_info.value
            file_chooser.set_files(file.name)

    # Add signal names
    modal.get_by_role("textbox", name="Signal Names").fill("ip, dalpha")

    # Create samples
    modal.get_by_role("button", name="Add Samples").click()

    # Check samples added to table
    expect(page.get_by_role("row").nth(1)).to_contain_text("12380")
    expect(page.get_by_role("row").nth(6)).to_contain_text("12385")

    # Check 6 samples added (12380 to 12385 inclusive)
    response = requests.get(f"http://localhost:8002/projects/{project_id}/samples")
    samples = response.json()
    assert len(samples) == 6
    assert all(
        sorted(sample["data"]["signal_names"]) == ["dalpha", "ip"] for sample in samples
    )


@pytest.mark.parametrize(
    "file_type", ["CSV", "TSV", "PARQUET", "FEATHER", "JSON", "XLSX"]
)
def test_create_samples_file_data(server_setup, page: Page, file_type: str):
    # Create a project
    project_id = create_project("Test Project", "time-series", "tabular")

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press create button
    page.get_by_role("button", name="Add Samples", exact=True).click()

    # Check modal has opened
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(modal.get_by_role("heading", name="Add Samples to Project")).to_be_visible()

    # Create some files of given type
    with tempfile.TemporaryDirectory() as tempd:
        pathlib.Path(tempd).joinpath(f"10000.{file_type.lower()}").touch()
        pathlib.Path(tempd).joinpath(f"10001.{file_type.lower()}").touch()

        # Check we can see File Type, Directory Path, and Signal Names input fields
        expect(modal.get_by_role("combobox", name="File Type")).to_be_visible()
        expect(modal.get_by_role("textbox", name="Directory Path")).to_be_visible()
        expect(modal.get_by_role("textbox", name="Signal Names")).to_be_visible()

        # Select given file type in dropdown
        modal.get_by_role("button", name="File Type").click()
        expect(page.get_by_role("option", name=file_type, exact=True)).to_be_visible()
        page.get_by_role("option", name=file_type, exact=True).click()

        # Add temp dir as file path, check 2 files are found
        modal.get_by_role("textbox", name="Directory Path").fill(tempd)
        expect(
            modal.get_by_text("Found 2 files with shot IDs: 10000, 10001")
        ).to_be_visible()

        # Add signal names
        modal.get_by_role("textbox", name="Signal Names").fill("ip, dalpha")

        # Create samples
        modal.get_by_role("button", name="Add Samples").click()

        # Check sample added to table
        expect(page.get_by_role("row").nth(1)).to_contain_text("10000")
        expect(page.get_by_role("row").nth(2)).to_contain_text("10001")

        # Check 2 samples added (10000 and 10001)
        response = requests.get(f"http://localhost:8002/projects/{project_id}/samples")
        samples = response.json()
        assert len(samples) == 2
        assert all(
            sorted(sample["data"]["signal_names"]) == ["dalpha", "ip"]
            for sample in samples
        )
        assert sorted(sample["shot_id"] for sample in samples) == [10000, 10001]


@pytest.mark.parametrize("file_type", ["PNG", "JPEG"])
def test_create_samples_image_data(server_setup, page: Page, file_type: str):
    # Create a project
    project_id = create_project("Test Project", "video", "image")

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press create button
    page.get_by_role("button", name="Add Samples", exact=True).click()

    # Check modal has opened
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(modal.get_by_role("heading", name="Add Samples to Project")).to_be_visible()

    # Create some image files
    with tempfile.TemporaryDirectory() as tempd:
        for shot_id in ("10000", "10001"):
            pathlib.Path(tempd).joinpath(shot_id).mkdir()
            pathlib.Path(tempd).joinpath(shot_id, "101.png").touch()
            pathlib.Path(tempd).joinpath(shot_id, "102.png").touch()
            pathlib.Path(tempd).joinpath(shot_id, "101.jpeg").touch()
            pathlib.Path(tempd).joinpath(shot_id, "102.jpeg").touch()

        # Check we can see File Type, File Path, and NOT File Columns
        expect(modal.get_by_role("combobox", name="File Type")).to_be_visible()
        expect(modal.get_by_role("textbox", name="Directory Path")).to_be_visible()

        # Choose the relevant file type
        modal.get_by_role("button", name="File Type").click()
        page.get_by_role("option", name=file_type).click()

        # Add temp dir as file path, check 2 directories are found
        modal.get_by_role("textbox", name="Directory Path").fill(tempd)
        expect(
            modal.get_by_text("Found 2 directories with shot IDs: 10000, 10001")
        ).to_be_visible()

        # Create samples
        modal.get_by_role("button", name="Add Samples").click()

        # Check samples added to table
        expect(page.get_by_role("row").nth(1)).to_contain_text("10000")
        expect(page.get_by_role("row").nth(2)).to_contain_text("10001")


def test_clear_samples(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And samples
    create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[1], ["Ip"]
    )
    # Navigate to samples page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press clear samples button
    page.get_by_role("button", name="Clear Samples").click()

    # Check warning modal opens
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()
    expect(
        modal.get_by_role("heading", name="Confirm Clear All Samples")
    ).to_be_visible()

    expect(
        modal.get_by_text(
            "Are you sure you want to delete all samples in this project?"
        )
    ).to_be_visible()

    expect(modal.get_by_role("button", name="Cancel")).to_be_visible()
    expect(modal.get_by_role("button", name="Clear All")).to_be_visible()

    # Press cancel, check samples still exist
    modal.get_by_role("button", name="Cancel").click()

    expect(page.get_by_role("row").nth(1)).to_contain_text("10000")
    expect(page.get_by_role("row").nth(2)).to_contain_text("10001")

    response = requests.get(f"http://localhost:8002/projects/{project_id}/samples")
    samples = response.json()
    assert len(samples) == 2

    # Press clear samples
    page.get_by_role("button", name="Clear Samples").click()

    # Check warning modal opens
    modal = page.get_by_role("dialog")
    expect(modal).to_be_visible()

    # Press clear, check samples still exist
    modal.get_by_role("button", name="Clear All").click()

    expect(page.get_by_role("row").nth(1)).to_be_hidden()

    response = requests.get(f"http://localhost:8002/projects/{project_id}/samples")
    samples = response.json()
    assert len(samples) == 0


@pytest.mark.parametrize("sample_id", (True, False))
def test_samples_page_import_annotations(sample_id: bool, server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And a sample
    sample_ids = create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    # Navigate to projects page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Create a Time Point annotation using sample ID in schema
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
        annotations_1 = [
            {
                "label": "Disruption",
                "created_by": "manual",
                "time": 71,
                "shot_id": 10000,
            },
            {
                "label": "Flat Top",
                "created_by": "manual",
                "time_min": 50,
                "time_max": 70,
                "shot_id": 10000,
            },
        ]
        annotations_2 = [
            {
                "label": "Control Loss",
                "created_by": "manual",
                "time": 61,
                "shot_id": 10001,
            },
            {
                "label": "Ramp Up",
                "created_by": "manual",
                "time_min": 40,
                "time_max": 60,
                "shot_id": 10001,
            },
        ]
        if sample_id:
            for annotation in annotations_1:
                annotation["sample_id"] = sample_ids[0]
            for annotation in annotations_2:
                annotation["sample_id"] = sample_ids[1]

        json.dump(annotations_1 + annotations_2, file)
        file.flush()
        # Make sure file is written
        time.sleep(0.5)
        # Import annotation
        with page.expect_file_chooser() as fc_info:
            page.get_by_role("button", name="Import Annotations").click()
            file_chooser = fc_info.value
            file_chooser.set_files(file.name)

        # Navigate to first sample, check annotations visible
        page.goto(
            f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}"
        )

        expect(page.get_by_role("gridcell", name="Disruption")).to_be_visible()
        expect(page.get_by_role("gridcell", name="Flat Top")).to_be_visible()

        expect(page.get_by_label("time-zone", exact=True)).to_have_count(1)
        expect(page.get_by_label("time-point", exact=True)).to_have_count(1)

        # Navigate to second sample, check annotations visble
        page.goto(
            f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[1]}"
        )

        expect(page.get_by_role("gridcell", name="Control Loss")).to_be_visible()
        expect(page.get_by_role("gridcell", name="Ramp Up")).to_be_visible()

        expect(page.get_by_label("time-zone", exact=True)).to_have_count(1)
        expect(page.get_by_label("time-point", exact=True)).to_have_count(1)


def test_samples_page_export_annotations(server_setup, page: Page):
    # Create project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And a sample
    sample_ids = create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    # Add annotations
    flat_top = TimeRegion(
        label="Flat Top", created_by="peak_detection", time_min=50, time_max=70
    )
    disruption = TimePoint(label="Disruption", created_by="peak_detection", time=71)
    response = requests.put(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_ids[0]}/annotations",
        json=[model.model_dump(mode="json") for model in (flat_top, disruption)],
    )
    assert response.status_code == 200

    ramp_up = TimeRegion(
        label="Ramp Up", created_by="peak_detection", time_min=40, time_max=60
    )
    control_loss = TimePoint(label="Control Loss", created_by="peak_detection", time=61)
    response = requests.put(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_ids[1]}/annotations",
        json=[model.model_dump(mode="json") for model in (ramp_up, control_loss)],
    )

    assert response.status_code == 200

    # Navigate to projects page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Press export annotations
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Export Annotations").click()

        download = download_info.value
        with tempfile.TemporaryDirectory() as tempd:
            download.save_as(pathlib.Path(tempd).joinpath("annotations.json"))

            with open(pathlib.Path(tempd).joinpath("annotations.json"), "r") as file:
                annotations = json.load(file)

    # Get flat top annotation
    exported_flat_top = next(ann for ann in annotations if ann["label"] == "Flat Top")
    # Check values are correct
    assert exported_flat_top["time_min"] == 50
    assert exported_flat_top["time_max"] == 70
    assert exported_flat_top["shot_id"] == 10000
    assert exported_flat_top["sample_id"] == sample_ids[0]

    # Get disruption annotation
    exported_disruption = next(
        ann for ann in annotations if ann["label"] == "Disruption"
    )
    # Check values are correct
    assert exported_disruption["time"] == 71
    assert exported_disruption["shot_id"] == 10000
    assert exported_disruption["sample_id"] == sample_ids[0]

    # Get ramp up annotation
    exported_ramp_up = next(ann for ann in annotations if ann["label"] == "Ramp Up")
    # Check values are correct
    assert exported_ramp_up["time_min"] == 40
    assert exported_ramp_up["time_max"] == 60
    assert exported_ramp_up["shot_id"] == 10001
    assert exported_ramp_up["sample_id"] == sample_ids[1]

    # Get control loss annotation
    exported_control_loss = next(
        ann for ann in annotations if ann["label"] == "Control Loss"
    )
    # Check values are correct
    assert exported_control_loss["time"] == 61
    assert exported_control_loss["shot_id"] == 10001
    assert exported_control_loss["sample_id"] == sample_ids[1]


@pytest.mark.models_enabled
@pytest.mark.parametrize(
    "model_name", ["mock_timeseries_cnn", "mock_params_timeseries_cnn"]
)
def test_model_train_predict(server_setup, setup_model_samples, page: Page, model_name):
    project_id, sample_ids = create_model_samples(setup_model_samples)

    # Navigate to projects page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Check contextual help explains models buttons

    # Click on model train modal
    page.get_by_role("button", name="Train ML Model").click()

    # Check modal has opened
    expect(page.get_by_role("heading", name="Train ML Model")).to_be_visible()
    expect(page.get_by_role("combobox", name="Select Model Type")).to_be_visible()
    expect(page.get_by_role("button", name="Close")).to_be_visible()
    expect(page.get_by_role("button", name="Train", exact=True)).to_be_visible()
    expect(page.get_by_role("switch", name="Allocate GPU")).to_be_visible()

    # Click on dropdown box, check models are shown
    page.get_by_role("button", name="Select Model Type").click()
    expect(
        page.get_by_role("option", name="disruption_cnn", exact=True)
    ).to_be_visible()
    expect(
        page.get_by_role("option", name="mock_timeseries_cnn", exact=True)
    ).to_be_visible()
    expect(
        page.get_by_role("option", name="mock_params_timeseries_cnn", exact=True)
    ).to_be_visible()

    page.get_by_role("option", name=model_name, exact=True).click()

    # If params model chosen, new form should open
    if model_name == "mock_params_timeseries_cnn":
        form_check(page, "Train")

    # Click train, should get accepted message
    page.get_by_role("button", name="Train", exact=True).click()
    expect(page.get_by_text("Model training added to job queue!")).to_be_visible()

    # Close modal, check it disappears
    page.get_by_role("button", name="Close", exact=True).click()

    expect(page.get_by_role("heading", name="Train ML Model")).to_be_hidden()
    expect(page.get_by_role("combobox", name="Select Model Type")).to_be_hidden()
    expect(page.get_by_role("button", name="Close")).to_be_hidden()
    expect(page.get_by_role("button", name="Train", exact=True)).to_be_hidden()

    # Open predict modal, check structure is correct
    page.get_by_role("button", name="Create Predictions from ML Model").click()
    modal = page.get_by_role("dialog", name="Create Predictions from ML Model")

    expect(
        page.get_by_role("heading", name="Create Predictions from ML Model")
    ).to_be_visible()
    expect(modal.get_by_role("textbox", name="Number of Predictions")).to_be_visible()
    expect(
        modal.get_by_role("button", name="Cancel Training", exact=True)
    ).to_be_visible()
    expect(modal.get_by_role("button", name="Predict", exact=True)).to_be_visible()
    expect(modal.get_by_role("button", name="Predict", exact=True)).to_be_disabled()
    expect(modal.get_by_role("button", name="Close", exact=True)).to_be_visible()

    # Check entry is there for newly trained model
    expect(modal.get_by_role("row").nth(1)).to_contain_text(model_name)
    expect(modal.get_by_role("row").nth(1)).to_contain_text("completed", timeout=60000)
    if model_name == "mock_params_timeseries_cnn":
        expect(modal.get_by_role("row").nth(1)).to_contain_text("50")
    else:
        expect(modal.get_by_role("row").nth(1)).to_contain_text("60")

    # Check cancel training button disabled after training complete
    expect(
        modal.get_by_role("button", name="Cancel Training", exact=True)
    ).to_be_disabled()

    # Click to select 10 predicions
    modal.get_by_role("button", name="Decrease Number of Predictions").click()
    expect(modal.get_by_role("textbox", name="Number of Predictions")).to_have_value(
        "10"
    )

    # Select our model from the list
    modal.get_by_role("checkbox", name=f"Select {model_name}").click()

    # If params model chosen, new form should open
    if model_name == "mock_params_timeseries_cnn":
        form_check(page, "Predict")

    # Check Predict button has been enabled, click it
    expect(modal.get_by_role("button", name="Predict", exact=True)).to_be_enabled()
    modal.get_by_role("button", name="Predict", exact=True).click()

    # Check message is shown
    expect(page.get_by_text("Model predictions added to job queue!")).to_be_visible()

    # Close the modal, check it closes
    modal.get_by_role("button", name="Close", exact=True).click()
    expect(
        page.get_by_role("heading", name="Create Predictions from ML Model")
    ).to_be_hidden()

    # Wait for a short time
    time.sleep(1)

    # Check 10 * 3 non-validated predictions added
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/annotations?validated=False",
    )
    assert response.status_code == 200
    annotations = response.json()
    assert len(annotations) == 30

    if model_name == "mock_params_timeseries_cnn":
        # Check disruptions have the value of params.final_score + 1
        assert all(
            ann["time"] == 51 for ann in annotations if ann["label"] == "Disruption"
        )
    else:
        # Hardcoded time to 60+1 inside mock model
        assert all(
            ann["time"] == 61 for ann in annotations if ann["label"] == "Disruption"
        )


@pytest.mark.models_enabled
def test_model_load_predict(server_setup, setup_model_samples, page: Page):
    project_id, sample_ids = create_model_samples(setup_model_samples)
    with tempfile.NamedTemporaryFile(suffix=".model", mode="w") as tempf:
        tempf.write("Loaded Model Weights")
        tempf.flush()

        # Navigate to projects page
        page.goto(f"http://localhost:8002/ui/projects/{project_id}")

        # Check basic structure of page is correct
        check_base_page(page)

        # Click on model train modal
        page.get_by_role("button", name="Load ML Model").click()

        # Check modal has opened
        expect(
            page.get_by_role("heading", name="Load Pretrained Model Weights")
        ).to_be_visible()
        expect(page.get_by_role("combobox", name="Select Model Type")).to_be_visible()
        expect(page.get_by_role("button", name="Close")).to_be_visible()
        expect(page.get_by_role("button", name="Submit", exact=True)).to_be_visible()

        # Check it has opened in local file mode
        expect(page.get_by_role("tab", name="Use Local File")).to_be_visible()
        expect(page.get_by_role("textbox", name="Model Weights Path")).to_be_visible()

        # Click on dropdown box, check models are shown
        page.get_by_role("button", name="Select Model Type").click()
        expect(
            page.get_by_role("option", name="disruption_cnn", exact=True)
        ).to_be_visible()
        expect(
            page.get_by_role("option", name="mock_timeseries_cnn", exact=True)
        ).to_be_visible()
        expect(
            page.get_by_role("option", name="mock_params_timeseries_cnn", exact=True)
        ).to_be_visible()

        page.get_by_role(
            "option", name="mock_params_timeseries_cnn", exact=True
        ).click()

        # Enter path to weights
        page.get_by_role("textbox", name="Model Weights Path").fill(tempf.name)

        # Click submit, should get loading wheel and then success message
        page.get_by_role("button", name="Submit", exact=True).click()
        expect(page.get_by_text("Model loaded successfully!")).to_be_visible(
            timeout=30000
        )
        # Close modal, check it disappears
        page.get_by_role("button", name="Close", exact=True).click()

        expect(
            page.get_by_role("heading", name="Load Pretrained Model Model")
        ).to_be_hidden()
        expect(page.get_by_role("combobox", name="Select Model Type")).to_be_hidden()
        expect(page.get_by_role("button", name="Close")).to_be_hidden()
        expect(page.get_by_role("button", name="Submit", exact=True)).to_be_hidden()

        # Check model has had a copy of the weights file added to cache
        response = requests.get(
            f"http://localhost:8002/projects/{project_id}/models/mock_params_timeseries_cnn",
        )
        assert response.status_code == 200
        # Get model ID
        model_id = response.json()["_id"]
        # Check file exists and has correct contents
        new_weights_path = config.settings.models.cache_dir.joinpath(
            f"{model_id}.model"
        )
        assert new_weights_path.exists()
        assert new_weights_path.read_text() == "Loaded Model Weights"

        # Check temp copy not left dangling
        file_name = pathlib.Path(tempf.name).name
        assert not config.settings.models.cache_dir.joinpath(file_name).exists()

        # Open predict modal, check structure is correct
        page.get_by_role("button", name="Create Predictions from ML Model").click()
        modal = page.get_by_role("dialog", name="Create Predictions from ML Model")

        expect(
            page.get_by_role("heading", name="Create Predictions from ML Model")
        ).to_be_visible()
        expect(
            modal.get_by_role("textbox", name="Number of Predictions")
        ).to_be_visible()
        expect(
            modal.get_by_role("button", name="Cancel Training", exact=True)
        ).to_be_visible()
        expect(modal.get_by_role("button", name="Predict", exact=True)).to_be_visible()
        expect(modal.get_by_role("button", name="Predict", exact=True)).to_be_disabled()
        expect(modal.get_by_role("button", name="Close", exact=True)).to_be_visible()
        expect(page.get_by_role("switch", name="Allocate GPU")).to_be_visible()

        # Check entry is there for newly trained model
        expect(modal.get_by_role("row").nth(1)).to_contain_text(
            "mock_params_timeseries_cnn"
        )
        expect(modal.get_by_role("row").nth(1)).to_contain_text("completed")

        # Check cancel training button disabled after loading complete
        expect(
            modal.get_by_role("button", name="Cancel Training", exact=True)
        ).to_be_disabled()

        # Click to select 10 predicions
        modal.get_by_role("button", name="Decrease Number of Predictions").click()
        expect(
            modal.get_by_role("textbox", name="Number of Predictions")
        ).to_have_value("10")

        # Select our model from the list
        modal.get_by_role("checkbox", name="Select mock_params_timeseries_cnn").click()

        # If params model chosen, new form should open
        form_check(page, "Predict")

        # Check Predict button has been enabled, click it
        expect(modal.get_by_role("button", name="Predict", exact=True)).to_be_enabled()
        modal.get_by_role("button", name="Predict", exact=True).click()

        # Check message is shown
        expect(
            page.get_by_text("Model predictions added to job queue!")
        ).to_be_visible()

        # Close the modal, check it closes
        modal.get_by_role("button", name="Close", exact=True).click()
        expect(
            page.get_by_role("heading", name="Create Predictions from ML Model")
        ).to_be_hidden()

        # Wait for a short time
        time.sleep(1)

        # Check 10 * 3 non-validated predictions added
        response = requests.get(
            f"http://localhost:8002/projects/{project_id}/annotations?validated=False",
        )
        assert response.status_code == 200
        annotations = response.json()
        assert len(annotations) == 30

        assert all(
            ann["time"] == 51 for ann in annotations if ann["label"] == "Disruption"
        )


@pytest.mark.models_disabled
def test_models_disabled(server_setup, page: Page):
    project_id = create_project("Test Project", "time-series", "tabular")

    # Navigate to projects page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Check model buttons disabled
    expect(
        page.get_by_role("button", name="Create Predictions from ML Model")
    ).to_be_disabled()
    expect(page.get_by_role("button", name="Train ML Model")).to_be_disabled()
    expect(page.get_by_role("button", name="Load ML Model")).to_be_disabled()

    # Open contextual help, check it contains correct info
    expect(page.get_by_role("button", name="ML Model Help")).to_be_visible()
    page.get_by_role("button", name="ML Model Help").click()

    expect(page.get_by_text("ML Models Disabled")).to_be_visible()
