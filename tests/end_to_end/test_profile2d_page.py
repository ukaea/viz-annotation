import pathlib
import time
from typing import Callable, Literal, Tuple

import requests
from playwright.sync_api import Page, expect

from tests.endpoints import create_local_samples, create_project

# The id of the div the Profile2D plot renders into (see base-plot.tsx).
PLOT_ID = "Profile2DView"


def setup_project(page: Page) -> Tuple[str, str, Callable]:
    project_id = create_project("Test Profile2D Project", "profile-2d", "tabular")
    ids = create_local_samples(
        project_id,
        [10000],
        pathlib.Path(__file__).parents[1],
        ["Ip"],
        file_names=["profile2d.parquet"],
    )
    sample_id = ids[0]

    page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}")

    # The signal is auto-selected, so the spectrogram renders without interaction.
    expect(page.get_by_label("profile-2d")).to_be_visible()

    def reload(project_id=project_id, sample_id=sample_id):
        page.goto(f"http://localhost:8002/ui/projects/{project_id}/samples/{sample_id}")

    return (project_id, sample_id, reload)


def get_dragmode(page: Page) -> str:
    return page.evaluate(
        """(plotId) => {
            const plot = document.getElementById(plotId);
            return plot && plot._fullLayout ? plot._fullLayout.dragmode : null;
        }""",
        PLOT_ID,
    )


def add_bounding_box(page: Page, label: str = "NTM", offset: int = 150) -> None:
    # Switch into edit mode (the button is labelled with the CURRENT mode) and pick
    # the bounding box tool + a label, then Ctrl-drag a box on the heatmap subplot.
    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()
    page.get_by_role("button", name="BOUNDING BOX").click()
    page.get_by_test_id("select-annotation-label").click()
    page.get_by_test_id("popover").get_by_text(label).click()

    box = page.get_by_label("profile-2d").bounding_box()
    assert box is not None
    # Start in the upper-left of the heatmap (yaxis2 spans the top 80% of the plot).
    start_x = box["x"] + box["width"] * 0.35
    start_y = box["y"] + box["height"] * 0.35

    page.mouse.move(start_x, start_y)
    page.keyboard.down("Control")
    page.mouse.down()
    page.mouse.move(start_x + offset, start_y + offset, steps=20)
    page.mouse.up()
    page.keyboard.up("Control")

    page.get_by_role("button", name="Edit Mode").click()
    expect(page.get_by_label("bounding-box")).to_have_count(1)


def add_time_annotation(
    page: Page,
    annotation_type: Literal["TIME REGION", "TIME POINT"],
    label: str,
    offset: int = 150,
) -> None:
    # Time Region/Time Point are not restricted to the heatmap subplot, so (unlike
    # add_bounding_box) the drag can start anywhere on the full plot.
    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()
    page.get_by_role("button", name=annotation_type).click()
    page.get_by_test_id("select-annotation-label").click()
    page.get_by_test_id("popover").get_by_text(label).click()

    box = page.get_by_label("profile-2d").bounding_box()
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


def test_profile2d_plot_renders(server_setup, page: Page):
    setup_project(page)
    expect(page.get_by_label("profile-2d")).to_be_visible()
    # Default drag mode is pan (matching the time series view) so left-drag pans.
    assert get_dragmode(page) == "pan"


def test_profile2d_threshold_toggle_preserves_manual_annotations(
    server_setup, page: Page
):
    """Toggling thresholding must not clear the user's own annotations."""
    setup_project(page)

    add_bounding_box(page)

    # Enabling thresholding adds polygons but must keep the manual annotation.
    page.get_by_role("button", name="Threshold").click()
    page.get_by_role("switch", name="Thresholding").click(force=True)
    expect(page.get_by_label("polygon").first).to_be_visible()
    expect(page.get_by_label("bounding-box")).to_have_count(1)

    # Disabling removes the generated polygons but keeps the manual annotation.
    page.get_by_role("switch", name="Thresholding").click(force=True)
    expect(page.get_by_label("polygon")).to_have_count(0)
    expect(page.get_by_label("bounding-box")).to_have_count(1)


def test_profile2d_saved_threshold_annotations_persist(server_setup, page: Page):
    """Saving threshold annotations validates them, so toggling the annotator off
    no longer discards them."""
    project_id, sample_id, _reload = setup_project(page)

    page.get_by_role("button", name="Threshold").click()
    page.get_by_role("switch", name="Thresholding").click(force=True)
    expect(page.get_by_label("polygon").first).to_be_visible()
    enabled_count = page.get_by_label("polygon").count()

    # Save, waiting for the PUT to the backend to complete.
    with page.expect_response(
        lambda r: (
            f"samples/{sample_id}/annotations" in r.url and r.request.method == "PUT"
        )
    ):
        page.get_by_role("button", name="Save").click(force=True)

    # Toggle the annotator off - the saved annotations must remain on the plot.
    page.get_by_role("switch", name="Thresholding").click(force=True)
    expect(page.get_by_label("polygon")).to_have_count(enabled_count)

    # They should be stored as validated, while keeping the annotator that produced
    # them as their creator rather than being reattributed to the user.
    annotations = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    ).json()
    assert len(annotations) >= 1
    for annotation in annotations:
        assert annotation["validated"]
        assert annotation["created_by"] == "profile_2d_threshold"


def test_profile2d_tools_disabled_in_view_mode(server_setup, page: Page):
    """Annotation tools can't be activated - and so nothing can be drawn - in View Mode."""
    setup_project(page)

    expect(page.get_by_role("button", name="View Mode")).to_be_enabled()
    expect(page.get_by_role("button", name="TIME REGION")).to_be_disabled()
    expect(page.get_by_role("button", name="TIME POINT")).to_be_disabled()
    expect(page.get_by_role("button", name="BOUNDING BOX")).to_be_disabled()
    expect(page.get_by_role("button", name="POLYGON")).to_be_disabled()

    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()

    expect(page.get_by_role("button", name="Edit Mode")).to_be_enabled()
    expect(page.get_by_role("button", name="TIME REGION")).to_be_enabled()
    expect(page.get_by_role("button", name="TIME POINT")).to_be_enabled()
    expect(page.get_by_role("button", name="BOUNDING BOX")).to_be_enabled()
    expect(page.get_by_role("button", name="POLYGON")).to_be_enabled()


def test_profile2d_save_time_annotations(server_setup, page: Page):
    """Manually drawn time regions/points save as the correct annotation types, and
    show up in the annotations table."""
    project_id, sample_id, _reload = setup_project(page)

    add_time_annotation(page, "TIME REGION", "NTM")
    add_time_annotation(page, "TIME POINT", "Disruption", offset=-150)

    expect(page.get_by_role("gridcell", name="NTM")).to_be_visible()
    expect(page.get_by_role("gridcell", name="Disruption")).to_be_visible()

    # Save, waiting for the PUT to the backend to complete.
    with page.expect_response(
        lambda r: (
            f"samples/{sample_id}/annotations" in r.url and r.request.method == "PUT"
        )
    ):
        page.get_by_role("button", name="Save").click(force=True)

    annotations = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    ).json()
    assert len(annotations) == 2
    for annotation in annotations:
        assert annotation["created_by"] == "manual"
        assert annotation["validated"]

    time_region = next(a for a in annotations if a["type"] == "time_region")
    assert time_region["label"] == "NTM"

    time_point = next(a for a in annotations if a["type"] == "time_point")
    assert time_point["label"] == "Disruption"


def test_profile2d_annotations_locked_in_view_mode(server_setup, page: Page):
    """Selecting or deleting an annotation is an edit-mode-only action."""
    setup_project(page)

    add_time_annotation(page, "TIME REGION", "NTM")
    # add_time_annotation leaves us back in View Mode.

    # Annotations have pointer events disabled in View Mode, so a right-click can't
    # reach one to open its context menu.
    page.get_by_label("time-zone").first.click(button="right", force=True)
    expect(page.get_by_role("menuitem", name="Delete")).to_have_count(0)
    expect(page.get_by_label("time-zone")).to_have_count(1)


def test_profile2d_edit_mode_relabel_and_delete(server_setup, page: Page):
    """In Edit Mode, an annotation's label can be changed and it can be deleted."""
    setup_project(page)

    add_time_annotation(page, "TIME REGION", "NTM")
    expect(page.get_by_role("gridcell", name="NTM")).to_be_visible()

    # add_time_annotation leaves us in View Mode - switch back into Edit Mode.
    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()

    # Relabel via the context menu's "Set type" submenu.
    page.get_by_label("time-zone").first.click(button="right")
    expect(page.get_by_role("menuitem", name="Delete")).to_be_visible()
    page.get_by_role("menuitem", name="Set type").hover()
    page.get_by_role("menuitem", name="ELM", exact=True).click()

    expect(page.get_by_role("gridcell", name="ELM", exact=True)).to_be_visible()
    expect(page.get_by_role("gridcell", name="NTM", exact=True)).to_have_count(0)

    # Delete it.
    page.get_by_label("time-zone").first.click(button="right")
    expect(page.get_by_role("menuitem", name="Delete")).to_be_visible()
    page.get_by_role("menuitem", name="Delete").click(force=True)

    expect(page.get_by_label("time-zone")).to_have_count(0)
