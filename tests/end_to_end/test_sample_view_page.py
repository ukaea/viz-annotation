from playwright.sync_api import Page, expect
import pathlib
from tests.endpoints import (
    create_project,
    create_local_samples,
    create_uda_samples,
    create_query_strategy_samples,
)
import pytest
import tempfile
import json
from toktagger.api.schemas.annotations import TimePoint, TimeRegion
import requests
import time


def setup_annotations(page: Page, num_annotations: int, go_to_next: bool = False):
    # Create project
    project_id = create_project(
        "Test Project", "time-series", "tabular", query_strategy="sequential"
    )
    # And a sample
    sample_ids = create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    # If > 0 annotations, Add 1 pre-existing annotation
    if num_annotations > 0:
        flat_top = TimeRegion(
            label="Flat Top",
            created_by="peak_detection",
            time_min=10,
            time_max=20,
            validated=False,
            uncertainty=0.9,
        )
        response = requests.put(
            f"http://localhost:8002/projects/{project_id}/samples/{sample_ids[0]}/annotations",
            json=[flat_top.model_dump(mode="json")],
        )
        response = requests.put(
            f"http://localhost:8002/projects/{project_id}/samples/{sample_ids[1]}/annotations",
            json=[flat_top.model_dump(mode="json")],
        )

        assert response.status_code == 200

    # Navigate to sample page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}")

    time.sleep(1)

    # Check basic structure of page is correct
    check_base_page(page)

    # If 2 annotations, add a new annotation via UI
    if num_annotations == 2:
        page.get_by_role("button", name="View Mode").click()
        page.locator("body").click()
        page.get_by_role("button", name="TIME POINT").click()
        page.get_by_test_id("select-annotation-label").click()
        page.get_by_test_id("popover").get_by_text("Disruption").click()
        page.get_by_label("time-series").click(button="left", modifiers=["Control"])

    if go_to_next:
        # Go forwards, create new annotations
        page.get_by_role("button", name="Next").click()
        page.wait_for_timeout(200)

        if num_annotations == 2:
            page.get_by_role("button", name="View Mode").click()
            page.locator("body").click()
            page.get_by_role("button", name="TIME POINT").click()
            page.get_by_test_id("select-annotation-label").click()
            page.get_by_test_id("popover").get_by_text("Disruption").click()
            page.get_by_label("time-series").click(button="left", modifiers=["Control"])

    time.sleep(1)

    return page, project_id, sample_ids


def check_base_page(page):
    # Expect page is called TokTagger
    expect(page).to_have_title("TokTagger")

    # Expect breadcrumbs at the top to be visible, and showing Projects & Samples link
    expect(page.get_by_role("link", name="Projects")).to_be_visible()
    expect(page.get_by_role("link", name="Project: Test Project")).to_be_visible()

    # Expect toolbar on left hand side to contain appropriate buttons
    expect(page.get_by_text("Controls")).to_be_visible()
    expect(page.get_by_role("button", name="Save")).to_be_visible()
    expect(page.get_by_role("button", name="Next")).to_be_visible()
    expect(page.get_by_role("button", name="Previous")).to_be_visible()
    expect(page.get_by_role("button", name="Clear", exact=True)).to_be_visible()
    expect(page.get_by_role("checkbox", name="Save on Navigate")).to_be_visible()
    expect(page.get_by_role("searchbox", name="Jump to Shot")).to_be_visible()

    # Check export and import annotations dropdowns present
    expect(page.get_by_role("button", name="Export Annotations")).to_be_visible()
    page.get_by_role("button", name="Export Annotations").click()
    expect(page.get_by_role("group", name="Export Annotations")).to_be_visible()
    expect(page.get_by_role("combobox", name="Export")).to_be_visible()
    expect(page.get_by_role("button", name="Export", exact=True)).to_be_visible()
    page.get_by_role("button", name="Export Annotations").click()
    expect(page.get_by_role("group", name="Export Annotations")).to_be_hidden()

    expect(page.get_by_role("button", name="Import Annotations")).to_be_visible()
    page.get_by_role("button", name="Import Annotations").click()
    expect(page.get_by_role("group", name="Import Annotations")).to_be_visible()
    expect(page.get_by_role("button", name="Import", exact=True)).to_be_visible()
    page.get_by_role("button", name="Import Annotations").click()
    expect(page.get_by_role("group", name="Import Annotations")).to_be_hidden()


@pytest.mark.parametrize("data_loader", ["tabular", "uda"])
def test_timeseries_navigation(data_loader, request, server_setup, page: Page):
    # Create Project
    project_id = create_project("Test Project", "time-series", data_loader)
    # And a sample for time-series
    if data_loader == "uda":
        request.getfixturevalue("uda_test")
        ids = create_uda_samples(project_id, [10000], ["Ip"])
    else:
        ids = create_local_samples(
            project_id, [10000], pathlib.Path(__file__).parents[1], ["Ip"]
        )

    sample_id = ids[0]

    # Navigate to Samples page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Click on sample
    page.get_by_role("rowheader", name="10000").click()

    # Check I've navigated to the correct page
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}?sortColumn=shot_id&sortDirection=ascending",
        timeout=3000,
    )

    # Check basic structure of page is correct
    check_base_page(page)

    # Check time series plot rendered
    expect(page.get_by_label("time-series")).to_be_visible(
        timeout=60000
    )  # For UDA this might be slow...

    # Check Ip trace rendered
    expect(page.get_by_text("Ip", exact=True)).to_be_visible()

    # Check Annotations table rendered
    expect(page.get_by_role("columnheader", name="Category")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Type")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Data")).to_be_visible()

    # Check Toolbox rendered
    expect(page.get_by_text("Toolbox")).to_be_visible()

    # Check toolbox buttons exist, clicking them opens the relevent group, can then be closed
    for tool in (
        "Shot Labels",
        "Peak Detection",
        "Outlier Detection",
        "Change Point Detection",
        "Jump Detection",
    ):
        expect(page.get_by_role("button", name=tool)).to_be_visible()
        page.get_by_role("button", name=tool).click()
        expect(page.get_by_role("group", name=tool)).to_be_visible()
        page.get_by_role("button", name=tool).click()
        expect(page.get_by_role("group", name=tool)).to_be_hidden()


# @pytest.mark.parametrize("img_type", ["png", "jpeg"])
# def test_ufo_navigation(img_type, server_setup, page: Page):
#     # Create Project
#     project_id = create_project("Test Project", "video", "image")

#     ids = create_image_samples(
#         project_id,
#         10000,
#         pathlib.Path(__file__).parents[1].joinpath("mast_images"),
#         img_type,
#     )

#     sample_id = ids[0]

#     # Navigate to Samples page
#     page.goto(f"http://localhost:8002/ui/projects/{project_id}")

#     # Click on sample
#     page.get_by_role("rowheader", name="10000").click()

#     # Check I've navigated to the correct page
#     expect(page).to_have_url(
#         f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}",
#         timeout=3000,
#     )

#     # Check basic structure of page is correct
#     check_base_page(page)

#     # Check frame navigation present
#     expect(page.get_by_role("searchbox", name="Jump to Frame")).to_be_visible()
#     expect(page.get_by_text("Frame: 1")).to_be_visible()

#     # Check image displayed
#     expect(page.get_by_role("img")).to_be_visible()


@pytest.mark.parametrize("data_loader", ["tabular", "uda"])
def test_search_for_shot(request, data_loader, server_setup, page: Page):
    # Create Project
    project_id = create_project("Test Project", "time-series", data_loader)
    # And a sample for time-series
    if data_loader == "uda":
        request.getfixturevalue("uda_test")
        shot_10000_id = create_uda_samples(project_id, [10000], ["Ip"])[0]
        shot_10001_id = create_uda_samples(project_id, [10001], ["Ip"])[0]
    else:
        shot_10000_id = create_local_samples(
            project_id, [10000], pathlib.Path(__file__).parents[1], ["Ip"]
        )[0]
        shot_10001_id = create_local_samples(
            project_id, [10001], pathlib.Path(__file__).parents[1]
        )[0]

    # Navigate to Samples page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Click on sample
    page.get_by_role("rowheader", name="10000").click()

    # Check I've navigated to the correct page
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{shot_10000_id}?sortColumn=shot_id&sortDirection=ascending",
        timeout=3000,
    )

    # Check basic structure of page is correct
    check_base_page(page)

    # Search by shot - go to 10001
    searchbox = page.get_by_role("searchbox", name="Jump to Shot")

    # Search for sample 10001
    searchbox.fill("10001")
    searchbox.press("Enter")

    # Check I've navigated to new page
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{shot_10001_id}?sortColumn=shot_id&sortDirection=ascending",
        timeout=3000,
    )

    # Check basic structure of page is correct
    check_base_page(page)

    # Try to navigate to non existent shot
    searchbox.fill("10002")
    searchbox.press("Enter")

    # Check I've not been moved off of the current page
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{shot_10001_id}?sortColumn=shot_id&sortDirection=ascending",
        timeout=3000,
    )
    # Check error message shown
    expect(page.get_by_text("Shot not found!")).to_be_visible()


def test_import_annotations(server_setup, page: Page):
    # Create a project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And a sample
    sample_ids = create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    # Navigate to samples page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Create a Time Point annotation using sample ID in schema
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
        annotations = [
            {
                "label": "Disruption",
                "created_by": "manual",
                "time": 71,
            },
            {
                "label": "Flat Top",
                "created_by": "manual",
                "time_min": 50,
                "time_max": 70,
            },
        ]

        json.dump(annotations, file)
        file.flush()
        # Make sure file is written
        time.sleep(0.5)

        # Expand Import Annotations group
        expect(page.get_by_role("button", name="Import Annotations")).to_be_visible()
        page.get_by_role("button", name="Import Annotations").click()
        expect(page.get_by_role("group", name="Import Annotations")).to_be_visible()

        # Import annotation
        with page.expect_file_chooser() as fc_info:
            page.get_by_role("button", name="Import", exact=True).click()
            file_chooser = fc_info.value
            file_chooser.set_files(file.name)

        # Check annotations visible
        expect(page.get_by_role("gridcell", name="Disruption")).to_be_visible()
        expect(page.get_by_role("gridcell", name="Flat Top")).to_be_visible()

        expect(page.get_by_label("time-zone", exact=True)).to_have_count(1)
        expect(page.get_by_label("time-point", exact=True)).to_have_count(1)


@pytest.mark.parametrize("all_samples", (True, False))
def test_export_annotations(server_setup, page: Page, all_samples: bool):
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

    # Navigate to sample page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}")

    # Check basic structure of page is correct
    check_base_page(page)

    # Expand Export Annotations dropdown
    expect(page.get_by_role("button", name="Export Annotations")).to_be_visible()
    page.get_by_role("button", name="Export Annotations").click()
    expect(page.get_by_role("group", name="Export Annotations")).to_be_visible()

    # Select either All or Current Sample
    page.get_by_role("button", name="Show suggestions Export").click()
    page.get_by_role(
        "option", name="All" if all_samples else "Current Sample", exact=True
    ).click()

    # Press export annotations
    with page.expect_download() as download_info:
        page.get_by_role("button", name="Export", exact=True).click()

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
    exported_ramp_up = next(
        (ann for ann in annotations if ann["label"] == "Ramp Up"), None
    )
    if all_samples:
        # Check values are correct
        assert exported_ramp_up["time_min"] == 40
        assert exported_ramp_up["time_max"] == 60
        assert exported_ramp_up["shot_id"] == 10001
        assert exported_ramp_up["sample_id"] == sample_ids[1]
    else:
        assert not exported_ramp_up

    # Get control loss annotation
    exported_control_loss = next(
        (ann for ann in annotations if ann["label"] == "Control Loss"), None
    )
    if all_samples:
        # Check values are correct
        assert exported_control_loss["time"] == 61
        assert exported_control_loss["shot_id"] == 10001
        assert exported_control_loss["sample_id"] == sample_ids[1]
    else:
        assert not exported_control_loss


@pytest.mark.parametrize("num_annotations", [0, 1, 2])
def test_save_button(server_setup, page: Page, num_annotations: int):
    # Create annotations
    page, project_id, sample_ids = setup_annotations(page, num_annotations)
    sample_id = sample_ids[0]

    # Click save
    with page.expect_response(
        lambda r: (
            f"samples/{sample_id}/annotations" in r.url and r.request.method == "PUT"
        )
    ):
        page.get_by_role("button", name="Save").click()

    # Check 'Saved annotations' text visible
    expect(page.get_by_text(f"Saved {num_annotations} annotations!")).to_be_visible()

    # Pull from backend, check correct number of annotations saved
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    )
    assert response.status_code == 200
    annotations = response.json()

    assert len(annotations) == num_annotations

    # Check all marked as validated:
    for annotation in annotations:
        assert annotation["validated"]

    # Check sample is now marked as validated
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}"
    )
    assert response.status_code == 200
    sample = response.json()
    assert sample["validated_annotations"]


@pytest.mark.parametrize("num_annotations", [0, 1, 2])
@pytest.mark.parametrize("save_on_navigate", [True, False])
@pytest.mark.parametrize("navigate_direction", ["Previous", "Next"])
def test_save_on_navigate(
    server_setup,
    page: Page,
    num_annotations: int,
    save_on_navigate: bool,
    navigate_direction: str,
):
    # Create annotations
    page, project_id, sample_ids = setup_annotations(
        page,
        num_annotations,
        go_to_next=True if navigate_direction == "Previous" else False,
    )
    sample_id = sample_ids[0] if navigate_direction == "Next" else sample_ids[1]

    # Disable Save on Navigate if required
    expect(page.get_by_role("checkbox", name="Save on Navigate")).to_be_checked()
    if not save_on_navigate:
        page.get_by_role("checkbox", name="Save on Navigate").click()
        expect(
            page.get_by_role("checkbox", name="Save on Navigate")
        ).not_to_be_checked()

    # Go to next/previous sample — wait for the save PUT to complete before proceeding
    if save_on_navigate:
        with page.expect_response(
            lambda r: (
                f"samples/{sample_id}/annotations" in r.url
                and r.request.method == "PUT"
            )
        ):
            page.get_by_role("button", name=f"{navigate_direction}").click()
    else:
        page.get_by_role("button", name=f"{navigate_direction}").click()

    # Check if annotations saved
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    )
    assert response.status_code == 200
    annotations = response.json()
    if save_on_navigate:
        assert len(annotations) == num_annotations
    else:
        assert len(annotations) == (
            0 if num_annotations == 0 else 1
        )  # Because it shouldnt have saved the human annotation if num_annotations=2

    # Check all marked as validated if saved
    for annotation in annotations:
        assert annotation["validated"] == (True if save_on_navigate else False)

    time.sleep(1)

    # Check sample is now marked as validated if saved
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}"
    )
    assert response.status_code == 200
    sample = response.json()
    assert sample["validated_annotations"] == (True if save_on_navigate else False)


def test_clear_button(server_setup, page: Page):
    page, project_id, sample_ids = setup_annotations(page, 2)
    sample_id = sample_ids[0]
    # Click save to commit annotations to db
    page.get_by_role("button", name="Save").click()

    # Check both annotations visible
    expect(page.get_by_label("time-point").first).to_be_visible()
    expect(page.get_by_label("time-zone").first).to_be_visible()

    # Check sample shows as validated
    expect(page.get_by_text("Annotations Validated")).to_be_visible()

    # Press Clear
    page.get_by_role("button", name="Clear").click()

    # Check no annotations visible
    expect(page.get_by_label("time-point").first).to_be_hidden()
    expect(page.get_by_label("time-zone").first).to_be_hidden()

    # Check sample now shows as not validated
    expect(page.get_by_text("Annotations Not Validated")).to_be_visible()

    # Press save, check no annotations in db
    page.get_by_role("button", name="Save").click()

    # Pull from backend, check correct number of annotations saved
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    )
    assert response.status_code == 200
    annotations = response.json()

    assert len(annotations) == 0


@pytest.mark.parametrize(
    ("query_strategy", "expected_order", "sort_by", "sort_direction"),
    [
        (
            "uncertainty",
            [
                0,
                4,
                2,
                3,
                1,
            ],  # Most uncertain, least uncertain, no annotations, validated annotations
            "shot_id",
            "ascending",
        ),
        (
            "uncertainty",
            [
                0,
                4,
                2,
                3,
                1,
            ],  # Most uncertain, least uncertain, no annotations, validated annotations
            "shot_id",
            "descending",  # Order of samples shouldn't make any difference
        ),
        (
            "random",
            [
                0,
                3,
                4,
                2,
                1,
            ],  # Random order of non-validated, then validated annotations
            "shot_id",
            "ascending",
        ),
        (
            "random",
            [
                0,
                3,
                2,
                4,
                1,
            ],  # Random order of non-validated, then validated annotations
            "shot_id",
            "descending",  # Different sample order, so different random order
        ),
        (
            "sequential",
            [0, 1, 2, 3, 4],  # Increasing order of shot
            "shot_id",
            "ascending",
        ),
        (
            "sequential",
            [2, 3, 4],  # Increasing order of shot, doesn't cycle back to earlier shots
            "shot_id",
            "ascending",
        ),
        (
            "sequential",
            [4, 3, 2, 1, 0],  # Decreasing order of shot
            "shot_id",
            "descending",
        ),
        (
            "sequential",
            [2, 1, 0],  # Decreasing order of shot, doesn't cycle back to later shots
            "shot_id",
            "descending",
        ),
        (
            "sequential",
            [
                4,
                3,
                2,
                1,
                0,
            ],  # Increasing order of time created (which is inverse of shot ID)
            "_id",
            "ascending",
        ),
        (
            "sequential",
            [
                2,
                1,
                0,
            ],  # Decreasing order of time created, doesn't cycle back to earlier shots
            "_id",
            "ascending",
        ),
        (
            "sequential",
            [
                0,
                1,
                2,
                3,
                4,
            ],  # Decreasing order of time created (which is inverse of shot ID)
            "_id",
            "descending",
        ),
        (
            "sequential",
            [
                2,
                3,
                4,
            ],  # Decreasing order of time created, doesn't cycle back to later shots
            "_id",
            "descending",
        ),
    ],
)
def test_query_strategies(
    server_setup, page: Page, query_strategy, expected_order, sort_by, sort_direction
):
    project_id, sample_ids = create_query_strategy_samples(
        query_strategy=query_strategy
    )

    # Go to shot starting shot, with given query params
    page.goto(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[expected_order[0]]}?sortColumn={sort_by}&sortDirection={sort_direction}"
    )

    # Press Next, check it gives the correct sample each time
    for i in range(1, len(expected_order)):
        page.get_by_role("button", name="Next").click()

        expect(page.get_by_label("time-series")).to_be_visible()

        # Check navigated to next sample
        expect(page).to_have_url(
            f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[expected_order[i]]}?sortColumn={sort_by}&sortDirection={sort_direction}"
        )

    # Press Next a final time, shouldn't navigate anywhere, should display 'No more samples'
    page.get_by_role("button", name="Next").click()
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[expected_order[-1]]}?sortColumn={sort_by}&sortDirection={sort_direction}"
    )
    expect(page.get_by_text("No more samples available")).to_be_visible()

    # Now navigate backwards with previous button, should do reverse order of visited
    for i in range(len(expected_order) - 1, 0, -1):
        page.get_by_role("button", name="Previous").click()

        expect(page.get_by_label("time-series")).to_be_visible()

        expect(page).to_have_url(
            f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[expected_order[i - 1]]}?sortColumn={sort_by}&sortDirection={sort_direction}"
        )

    # Should be back to start, previous button should be greyed out
    expect(page.get_by_role("button", name="Previous")).to_be_disabled()


def test_validated_alertbox(server_setup, page: Page):
    project_id, sample_ids = create_query_strategy_samples(query_strategy="sequential")
    # Go to sample with validated annotations
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[0]}")
    # Check box says they are validated
    expect(page.get_by_text("Annotations Validated")).to_be_visible()

    # Go to sample with non validated annotations
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[2]}")

    # Check box says they are NOT validated
    expect(page.get_by_text("Annotations Not Validated")).to_be_visible()

    # Press Save
    page.get_by_role("button", name="Save").click()

    # Check updates to validated
    expect(page.get_by_text("Annotations Validated")).to_be_visible()

    # Go to sample with no annotations
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[3]}")

    # Check box says they are NOT validated
    expect(page.get_by_text("Annotations Not Validated")).to_be_visible()

    # Disable save on navigate
    page.get_by_role("checkbox", name="Save on Navigate").click()
    expect(page.get_by_role("checkbox", name="Save on Navigate")).not_to_be_checked()

    # Press Next
    page.get_by_role("button", name="Next").click()

    # Check page updated
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[4]}"
    )

    # Go back
    page.get_by_role("button", name="Previous").click()

    # Check page updated
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[3]}"
    )

    # Check box says they are NOT validated since it wasn't saved
    expect(page.get_by_text("Annotations Not Validated")).to_be_visible()

    # Re-enable Save on Navigate
    page.get_by_role("checkbox", name="Save on Navigate").click()
    expect(page.get_by_role("checkbox", name="Save on Navigate")).to_be_checked()

    # Press Next
    page.get_by_role("button", name="Next").click()

    # Check page updated
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[4]}"
    )

    # This sample should be non validated
    expect(page.get_by_text("Annotations Not Validated")).to_be_visible()

    # Press Next, should not change sample since there are no more, but should update to validated
    page.get_by_role("button", name="Next").click()
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[4]}"
    )
    expect(page.get_by_text("Annotations Validated")).to_be_visible()

    # Go back to previous sample, check now it says Validated since it was saved on navigate before
    page.get_by_role("button", name="Previous").click()
    expect(page).to_have_url(
        f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_ids[3]}"
    )
    expect(page.get_by_text("Annotations Validated")).to_be_visible()
