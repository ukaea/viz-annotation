from playwright.sync_api import Page, expect
import pathlib
from tests.endpoints import create_project, create_local_samples, create_model_samples
import pytest
import requests
import time
from toktagger.api.schemas.annotations import TimePoint, TimeRegion
from typing import Literal, Tuple, Callable
from tests.end_to_end import form_check


def setup_project(page: Page) -> Tuple[str, str, Callable]:
    # Create Project
    project_id = create_project("Test Project", "time-series", "tabular")
    # And a sample for disruption
    ids = create_local_samples(
        project_id, [10000], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    sample_id = ids[0]

    # Navigate to page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}")

    # Check time series plot rendered
    expect(page.get_by_label("time-series")).to_be_visible()

    def reload(project_id=project_id, sample_id=sample_id):
        page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}")

    return (project_id, sample_id, reload)


def add_annotation(
    page: Page,
    annotation_type: Literal["TIME REGION", "TIME POINT"],
    label: str,
    offset: int = 200,
):
    # Begin zone tool
    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()
    page.get_by_role("button", name=annotation_type).click()
    page.get_by_test_id("select-annotation-label").click()
    page.get_by_test_id("popover").get_by_text(label).click()

    # Perform drag
    graph = page.get_by_label("time-series")
    box = graph.bounding_box()
    assert box is not None

    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2

    page.mouse.move(start_x, start_y)
    page.keyboard.down("Control")
    page.mouse.down()
    page.mouse.move(start_x + offset, start_y, steps=20)
    page.mouse.up()
    page.keyboard.up("Control")

    page.get_by_role("button", name="Edit Mode").click()

    time.sleep(1)


def test_annotation_toolbar(server_setup, page: Page):
    setup_project(page)

    # Check toolbar is visible
    expect(page.get_by_test_id("annotation-toolbar")).to_be_visible()

    # Check view mode
    expect(page.get_by_role("button", name="View Mode")).to_be_enabled()
    expect(page.get_by_role("button", name="TIME REGION")).to_be_disabled()
    expect(page.get_by_role("button", name="TIME POINT")).to_be_disabled()

    # Check edit mode
    page.get_by_role("button", name="View Mode").click()
    expect(page.get_by_label("annotation-context-help")).to_be_visible()

    page.locator("body").click()
    expect(page.get_by_role("button", name="Edit Mode")).to_be_enabled()
    expect(page.get_by_role("button", name="TIME REGION")).to_be_enabled()
    expect(page.get_by_role("button", name="TIME POINT")).to_be_enabled()

    # Check if label selection is hidden
    expect(page.get_by_test_id("select-annotation-label")).not_to_be_visible()

    page.get_by_role("button", name="TIME POINT").click()
    expect(page.get_by_role("button", name="TIME POINT")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(page.get_by_role("button", name="TIME REGION")).to_have_attribute(
        "aria-pressed", "false"
    )

    # Check label selection
    label_selector = page.get_by_test_id("select-annotation-label")
    expect(label_selector).to_be_visible()
    label_selector.click()

    time.sleep(1)

    visible_menu = page.get_by_test_id("popover")
    expect(visible_menu).to_be_visible()

    # Choose Add Time Region, check options load
    for item in (
        "Thermal Quench",
        "Current Quench",
        "Locked Mode",
        "VDE",
        "Control Loss",
        "Disruption",
    ):
        expect(visible_menu.get_by_text(item, exact=True)).to_be_visible()

    page.locator("body").click()

    page.get_by_role("button", name="TIME REGION").click()
    expect(page.get_by_role("button", name="TIME REGION")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(page.get_by_role("button", name="TIME POINT")).to_have_attribute(
        "aria-pressed", "false"
    )

    # Check label selection
    label_selector = page.get_by_test_id("select-annotation-label")
    expect(label_selector).to_be_visible()
    label_selector.click()

    visible_menu = page.get_by_test_id("popover")
    expect(visible_menu).to_be_visible()

    # Choose Add Time Region, check options load
    for item in (
        "ELM",
        "L-mode",
        "H-mode",
        "Thermal Quench",
        "Current Quench",
        "Sawtooth",
        "IRE",
        "Locked Mode",
        "VDE",
        "Flat Top",
        "Ramp Up",
        "Ramp Down",
    ):
        expect(visible_menu.get_by_text(item, exact=True)).to_be_visible()


@pytest.mark.parametrize("zone_type", ["Ramp Up", "Flat Top", "Ramp Down"])
def test_timeseries_add_time_zone(zone_type, server_setup, page: Page):
    setup_project(page)

    add_annotation(page, "TIME REGION", zone_type)

    # Check added to list
    expect(page.get_by_role("gridcell", name=zone_type)).to_be_visible()

    # Wait for a bit for Zone to fully render
    page.wait_for_timeout(500)

    page.get_by_role("button", name="View Mode").click()

    # Check you can right click to delete it
    page.get_by_label("time-zone").first.click(button="right")
    expect(page.get_by_role("menuitem", name="Delete")).to_be_visible()
    page.get_by_role("menuitem", name="Delete").click(force=True)

    # Wait for a bit for Zone to fully delete
    page.wait_for_timeout(500)

    page.get_by_role("button", name="Edit Mode").click()

    # Check it no longer exists
    expect(page.get_by_role("gridcell", name=zone_type, exact=True)).to_have_count(0)
    expect(page.get_by_label("time-zone").first).to_have_count(0)


@pytest.mark.parametrize("zone_type", ["Ramp Up", "Flat Top", "Ramp Down"])
@pytest.mark.parametrize("handle", ["leftHandle", "rightHandle"])
@pytest.mark.parametrize("drag_to", [".wdrag", ".edrag"])
def test_timeseries_drag_time_zone(
    zone_type, handle, drag_to, server_setup, page: Page
):
    setup_project(page)

    add_annotation(page, "TIME REGION", zone_type)

    # Check added to list
    expect(page.get_by_role("gridcell", name=zone_type)).to_be_visible()
    bounds_text = (
        page.get_by_role("row").nth(1).get_by_role("gridcell").nth(2).inner_text()
    )
    initial_left_position, initial_right_position = map(float, bounds_text.split(" - "))

    page.get_by_role("button", name="View Mode").click()

    # Click handle, drag to new position
    page.get_by_label(f"zone.{handle}").drag_to(page.locator(drag_to))
    time.sleep(0.1)
    # Check values in table correctly updated
    bounds_text = (
        page.get_by_role("row").nth(1).get_by_role("gridcell").nth(2).inner_text()
    )
    updated_left_position, updated_right_position = map(float, bounds_text.split(" - "))

    if drag_to == ".wdrag":
        # Dragging left, left position should have reduced
        assert updated_left_position < initial_left_position

        if handle == "leftHandle":
            # If left handle dragged, right position should be unchanged
            assert updated_right_position == initial_right_position
        else:
            # If right handle dragged, positions should 'swap', so right position should be where left position was previously
            assert updated_right_position == initial_left_position
    else:
        # Dragging right, right position should be the increased
        assert updated_right_position > initial_right_position

        if handle == "leftHandle":
            # If left handle dragged, positions should 'swap', so left position should be where right position was previously
            assert updated_left_position == initial_right_position
        else:
            # If right handle dragged, left position should be unchanged
            assert updated_left_position == initial_left_position


@pytest.mark.parametrize(
    "time_point_type",
    [
        "Thermal Quench",
        "Current Quench",
        "Locked Mode",
        "VDE",
        "Control Loss",
        "Disruption",
    ],
)
def test_timeseries_add_time_point(server_setup, page: Page, time_point_type: str):
    setup_project(page)

    add_annotation(page, "TIME POINT", time_point_type)

    # Check added to list
    expect(page.get_by_role("gridcell", name=time_point_type)).to_be_visible()

    # Wait for a bit for Zone to fully render
    page.wait_for_timeout(500)

    page.get_by_role("button", name="View Mode").click()

    # Check you can right click to delete it
    page.get_by_label("time-point").first.click(button="right")
    expect(page.get_by_role("menuitem", name="Delete")).to_be_visible()
    page.get_by_role("menuitem", name="Delete").click(force=True)

    # Wait for a bit for Zone to fully delete
    page.wait_for_timeout(500)

    # Check it no longer exists
    expect(
        page.get_by_role("gridcell", name=time_point_type, exact=True)
    ).to_have_count(0)
    expect(page.get_by_label("time-point").first).to_have_count(0)


@pytest.mark.parametrize(
    "time_point_type",
    [
        "Thermal Quench",
        "Current Quench",
        "Locked Mode",
        "VDE",
        "Control Loss",
        "Disruption",
    ],
)
@pytest.mark.parametrize("drag_to", [".wdrag", ".edrag"])
def test_timeseries_drag_vspan(
    drag_to: str, time_point_type: str, server_setup, page: Page
):
    setup_project(page)

    add_annotation(page, "TIME POINT", time_point_type)

    # Check added to list
    expect(page.get_by_role("gridcell", name=time_point_type)).to_be_visible()
    initial_position = float(
        page.get_by_role("row").nth(1).get_by_role("gridcell").nth(2).inner_text()
    )

    page.get_by_role("button", name="View Mode").click()

    # Click handle, drag to new position
    page.get_by_label("time-point").drag_to(page.locator(drag_to))
    time.sleep(0.1)
    # Check values in table correctly updated
    updated_position = float(
        page.get_by_role("row").nth(1).get_by_role("gridcell").nth(2).inner_text()
    )

    if drag_to == ".wdrag":
        # Dragging left, position should have reduced
        assert updated_position < initial_position
    else:
        # Dragging right, position should be increased
        assert updated_position > initial_position


def test_timeseries_save_annotations(server_setup, page: Page):
    project_id, sample_id, _reload = setup_project(page)

    add_annotation(page, "TIME REGION", "Flat Top")

    # Check added to list
    expect(page.get_by_role("gridcell", name="Flat Top")).to_be_visible()
    bounds_text = (
        page.get_by_role("row").nth(1).get_by_role("gridcell").nth(2).inner_text()
    )
    time_zone_left_position, time_zone_right_position = map(
        float, bounds_text.split(" - ")
    )

    page.wait_for_timeout(500)

    add_annotation(page, "TIME POINT", "Disruption", -200)

    # Check added to list
    expect(page.get_by_role("gridcell", name="Disruption")).to_be_visible()
    disruption_position = float(
        page.get_by_role("row").nth(2).get_by_role("gridcell").nth(2).inner_text()
    )

    # Press Save and wait for the PUT request to the server to complete
    with page.expect_response(
        lambda r: (
            f"samples/{sample_id}/annotations" in r.url and r.request.method == "PUT"
        )
    ):
        page.get_by_role("button", name="Save").click(force=True)

    # Check annotation stored in db
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    )
    assert response.status_code == 200
    annotations = response.json()

    assert len(annotations) == 2

    for annotation in annotations:
        assert annotation["created_by"] == "manual"
        assert annotation["validated"]  # == True
        assert annotation["uncertainty"] == 0

    disruption_annotation = next(
        ann for ann in annotations if ann["label"] == "Disruption"
    )
    assert round(disruption_annotation["time"], 4) == float(disruption_position)
    assert disruption_annotation["type"] == "time_point"

    flattop_annotation = next(ann for ann in annotations if ann["label"] == "Flat Top")
    assert round(flattop_annotation["time_min"], 4) == float(time_zone_left_position)
    assert round(flattop_annotation["time_max"], 4) == float(time_zone_right_position)
    assert flattop_annotation["type"] == "time_region"


def test_timeseries_load_annotations(server_setup, page: Page):
    project_id, sample_id, reload = setup_project(page)

    # Create annotations of each type
    rampup = TimeRegion(
        label="Ramp Up", created_by="peak_detection", time_min=10, time_max=20
    )
    flattop = TimeRegion(
        label="Flat Top", created_by="peak_detection", time_min=20, time_max=70
    )
    rampdown = TimeRegion(
        label="Ramp Down", created_by="peak_detection", time_min=70, time_max=90
    )
    disruption = TimePoint(label="Disruption", created_by="peak_detection", time=91)
    # Add annotations
    response = requests.put(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations",
        json=[
            model.model_dump(mode="json")
            for model in (rampup, flattop, rampdown, disruption)
        ],
    )
    assert response.status_code == 200

    reload()

    # One vspan and 3 zones visible
    expect(page.get_by_label("time-point").first).to_be_visible()
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(3)

    # Check all four entries have correct info in table
    row = page.get_by_role("row").filter(
        has=page.get_by_role("gridcell", name="Disruption")
    )
    assert float(row.get_by_role("gridcell").nth(2).inner_text()) == disruption.time

    row = page.get_by_role("row").filter(
        has=page.get_by_role("gridcell", name="Ramp Up")
    )
    bounds_text = row.get_by_role("gridcell").nth(2).inner_text()
    left_position, right_position = map(float, bounds_text.split(" - "))
    assert float(left_position) == rampup.time_min
    assert float(right_position) == rampup.time_max

    row = page.get_by_role("row").filter(
        has=page.get_by_role("gridcell", name="Flat Top")
    )
    bounds_text = row.get_by_role("gridcell").nth(2).inner_text()
    left_position, right_position = map(float, bounds_text.split(" - "))
    assert float(left_position) == flattop.time_min
    assert float(right_position) == flattop.time_max

    row = page.get_by_role("row").filter(
        has=page.get_by_role("gridcell", name="Ramp Down")
    )
    bounds_text = row.get_by_role("gridcell").nth(2).inner_text()
    left_position, right_position = map(float, bounds_text.split(" - "))
    assert float(left_position) == rampdown.time_min
    assert float(right_position) == rampdown.time_max


def test_timeseries_update_annotations(server_setup, page: Page):
    project_id, sample_id, reload = setup_project(page)

    # Create annotations of each type
    rampup = TimeRegion(
        label="Ramp Up", created_by="peak_detection", time_min=10, time_max=20
    )
    flattop = TimeRegion(
        label="Flat Top", created_by="peak_detection", time_min=20, time_max=70
    )
    disruption = TimePoint(label="Disruption", created_by="peak_detection", time=71)
    # Add annotations
    response = requests.put(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations",
        json=[model.model_dump(mode="json") for model in (rampup, flattop, disruption)],
    )
    assert response.status_code == 200

    reload()

    # One vspan and 2 zones visible
    expect(page.get_by_label("time-point").first).to_be_visible()
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(2)

    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()

    # Delete a zone
    page.get_by_label("time-zone").first.click(button="right")
    expect(page.get_by_role("menuitem", name="Delete")).to_be_visible()
    page.get_by_role("menuitem", name="Delete").click(force=True)

    # Move the disruption
    # Click handle, drag to new position
    page.get_by_label("time-point").drag_to(page.locator(".edrag"))

    time.sleep(1)

    row = page.get_by_role("row").filter(
        has=page.get_by_role("gridcell", name="Disruption")
    )
    updated_disruption_time = float(row.get_by_role("gridcell").nth(2).inner_text())

    # Press Save and wait for the PUT request to the server to complete
    with page.expect_response(
        lambda r: (
            f"samples/{sample_id}/annotations" in r.url and r.request.method == "PUT"
        )
    ):
        page.get_by_role("button", name="Save").click(force=True)

    # Check annotation stored in db
    response = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    )
    assert response.status_code == 200
    annotations = response.json()

    time.sleep(1)

    # Check one annotation removed
    assert len(annotations) == 2
    print(annotations)

    # Check all annotations marked as validated
    for annotation in annotations:
        assert annotation["validated"]  # == True
        assert annotation["uncertainty"] == 0

    # Check time of disruption updated
    disruption_annotation = next(
        ann for ann in annotations if ann["label"] == "Disruption"
    )
    assert round(disruption_annotation["time"], 4) == updated_disruption_time


def test_timeseries_annotator(server_setup, page: Page):
    setup_project(page)

    # Expand find peaks annotator
    expect(page.get_by_role("button", name="Peak Detection")).to_be_visible()
    page.get_by_role("button", name="Peak Detection").click()
    peak_detection = page.get_by_role("group", name="Peak Detection")
    expect(peak_detection).to_be_visible()
    peak_detection.get_by_role("switch", name="Enable Tool").click()

    # Choose ip
    peak_detection.get_by_role("button", name="Show suggestions Signal Name").click()
    page.get_by_role("option", name="Ip").click()

    # Check that 4 peaks have been identified
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(4)

    # Disable tool, these should disappear
    peak_detection.get_by_role("switch", name="Enable Tool").click()
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(0)

    # Enable tool, they should reappear
    peak_detection.get_by_role("switch", name="Enable Tool").click()
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(4)

    # Change settings in toolbar, check they impact on annotations
    # Here drags min time to 74, so should only have one peak within window
    time_range = peak_detection.get_by_role("group", name="Time Range")
    min_slider = time_range.get_by_role("slider").nth(0)
    min_slider.scroll_into_view_if_needed()
    box = min_slider.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + 140, box["y"])  # drag horizontally
    page.mouse.up()

    # Check one annotation present
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(1)


@pytest.mark.models_enabled
@pytest.mark.parametrize(
    "model_name", ["mock_timeseries_cnn", "mock_params_timeseries_cnn"]
)
def test_timeseries_model_predict(
    server_setup, setup_model_samples, page: Page, model_name
):
    # Create Project
    project_id, sample_ids = create_model_samples(setup_model_samples)

    ids = create_local_samples(
        project_id, [10000], pathlib.Path(__file__).parents[1], ["Ip"]
    )

    sample_id = ids[0]

    # Navigate to samples table page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}")

    # Click on model train modal
    page.get_by_role("button", name="Train ML Model").click()

    # Check modal has opened
    expect(page.get_by_role("heading", name="Train ML Model")).to_be_visible()
    expect(page.get_by_role("combobox", name="Select Model Type")).to_be_visible()
    expect(page.get_by_role("button", name="Close")).to_be_visible()
    expect(page.get_by_role("button", name="Train", exact=True)).to_be_visible()
    expect(page.get_by_role("switch", name="Allocate GPU")).to_be_visible()

    # Click on dropdown box, check 'disruption_cnn' is shown
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
    expect(page.get_by_text("Model training added to job queue!")).to_be_visible(
        timeout=30000
    )

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

    # Check entry is there for newly trained model, wait for it to complete
    time.sleep(1)
    expect(modal.get_by_role("row").nth(1)).to_contain_text(model_name)
    expect(modal.get_by_role("row").nth(1)).to_contain_text("completed", timeout=30000)
    if model_name == "mock_params_timeseries_cnn":
        expect(modal.get_by_role("row").nth(1)).to_contain_text("50")
    else:
        expect(modal.get_by_role("row").nth(1)).to_contain_text("60")

    # Close modal
    modal.get_by_role("button", name="Close", exact=True).click()

    # Navigate to timeseries page
    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}")

    # Expand Model Predict in toolbar
    expect(page.get_by_role("button", name="Model Prediction")).to_be_visible()
    page.get_by_role("button", name="Model Prediction").click()
    model_predict = page.get_by_role("group", name="Model Prediction")
    expect(model_predict).to_be_visible()
    model_predict.get_by_role("switch", name="Enable Tool").click()
    expect(page.get_by_role("switch", name="Allocate GPU")).to_be_visible()

    # Choose model type
    model_predict.get_by_role(
        "combobox", name="Select Model Type"
    ).scroll_into_view_if_needed()
    model_predict.get_by_role(
        "button", name="Show suggestions Select Model Type"
    ).click()
    page.get_by_role("option", name=model_name, exact=True).click()

    # If params model chosen, new form should open
    if model_name == "mock_params_timeseries_cnn":
        form_check(page, "Predict")

    page.get_by_role("button", name="Predict", exact=True).click()

    # Should generate a new set of predictions after a short time
    expect(page.get_by_label("time-point", exact=True)).to_have_count(1)
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(2)

    # Check added to list
    expect(page.get_by_role("gridcell", name="Disruption")).to_be_visible()
    expect(page.get_by_role("gridcell", name="Ramp Up")).to_be_visible()
    expect(page.get_by_role("gridcell", name="Flat Top")).to_be_visible()

    row = page.get_by_role("row").filter(
        has=page.get_by_role("gridcell", name="Disruption")
    )
    if model_name == "mock_params_timeseries_cnn":
        # Check disruption has the value of params.final_score + 1
        assert float(row.get_by_role("gridcell").nth(2).inner_text()) == 51
    else:
        # Hardcoded time to 60+1 inside mock model
        assert float(row.get_by_role("gridcell").nth(2).inner_text()) == 61

    # Disable tool, it should disappear
    model_predict.get_by_role("switch", name="Enable Tool").click()
    expect(page.get_by_label("time-point", exact=True)).to_have_count(0)
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(0)

    expect(page.get_by_role("gridcell", name="Disruption")).to_be_hidden()
    expect(page.get_by_role("gridcell", name="Ramp Up")).to_be_hidden()
    expect(page.get_by_role("gridcell", name="Flat Top")).to_be_hidden()

    # Enable tool, it should reappear
    model_predict.get_by_role("switch", name="Enable Tool").click()
    expect(page.get_by_label("time-point", exact=True)).to_have_count(1)
    expect(page.get_by_label("time-zone", exact=True)).to_have_count(2)

    expect(page.get_by_role("gridcell", name="Disruption")).to_be_visible()
    expect(page.get_by_role("gridcell", name="Ramp Up")).to_be_visible()
    expect(page.get_by_role("gridcell", name="Flat Top")).to_be_visible()


@pytest.mark.models_disabled
def test_timeseries_models_disabled(server_setup, page: Page):
    setup_project(page)

    # Check model prediction tool is not visible
    expect(page.get_by_role("button", name="Model Prediction")).to_be_hidden()
