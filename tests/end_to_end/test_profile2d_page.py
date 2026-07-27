import pathlib
from typing import Callable, Tuple

import requests
from playwright.sync_api import Page, expect

from tests.endpoints import create_local_samples, create_project

# The Profile2D plot is rendered into a div with id "Profile2DView" (see
# base-plot.tsx) and aria-label "profile-2d" (profile2d.tsx). A profile-2d project
# backed by the `tabular` loader feeds the 1D signal through an STFT to produce the
# 2D spectrogram, so the offline parquet fixture is enough to exercise the view.
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


def get_x_range(page: Page) -> list[float]:
    """Read the live Plotly x-axis range from the rendered plot."""
    return page.evaluate(
        """(plotId) => {
            const plot = document.getElementById(plotId);
            const range = plot && plot._fullLayout && plot._fullLayout.xaxis
                ? plot._fullLayout.xaxis.range
                : null;
            return range ? [range[0], range[1]] : null;
        }""",
        PLOT_ID,
    )


def get_dragmode(page: Page) -> str:
    return page.evaluate(
        """(plotId) => {
            const plot = document.getElementById(plotId);
            return plot && plot._fullLayout ? plot._fullLayout.dragmode : null;
        }""",
        PLOT_ID,
    )


def _plot_centre(page: Page) -> tuple[float, float]:
    box = page.get_by_label("profile-2d").bounding_box()
    assert box is not None
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


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


def test_profile2d_plot_renders(server_setup, page: Page):
    setup_project(page)
    expect(page.get_by_label("profile-2d")).to_be_visible()
    # Default drag mode is pan (matching the time series view) so left-drag pans.
    page.wait_for_timeout(500)
    assert get_dragmode(page) == "pan"


def test_profile2d_wheel_zoom(server_setup, page: Page):
    """Problems 1 & 5: the mouse wheel should zoom in and gradually back out."""
    setup_project(page)
    page.wait_for_timeout(1000)

    cx, cy = _plot_centre(page)
    page.mouse.move(cx, cy)

    before = get_x_range(page)
    assert before is not None
    before_width = before[1] - before[0]

    # Scroll up to zoom in.
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(400)
    zoomed_in = get_x_range(page)
    zoomed_in_width = zoomed_in[1] - zoomed_in[0]
    assert zoomed_in_width < before_width, "wheel up should zoom the x-axis in"

    # Scroll down to zoom gradually back out (not a full reset).
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, 300)
    page.wait_for_timeout(400)
    zoomed_out = get_x_range(page)
    zoomed_out_width = zoomed_out[1] - zoomed_out[0]
    assert zoomed_out_width > zoomed_in_width, "wheel down should zoom the x-axis out"


def test_profile2d_zoom_preserved_after_annotation(server_setup, page: Page):
    """Problem 2: creating an annotation must not reset the zoom/viewport."""
    setup_project(page)
    page.wait_for_timeout(1000)

    cx, cy = _plot_centre(page)
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(400)

    before = get_x_range(page)
    assert before is not None
    before_width = before[1] - before[0]

    add_bounding_box(page)
    page.wait_for_timeout(1000)

    # The annotation should have been created...
    expect(page.get_by_role("gridcell", name="NTM").first).to_be_visible()

    # ...and the viewport should be unchanged (uirevision preserves it).
    after = get_x_range(page)
    after_width = after[1] - after[0]
    assert abs(after[0] - before[0]) <= 0.02 * before_width
    assert abs(after_width - before_width) <= 0.02 * before_width


def _annotation_shape_count(page: Page) -> int:
    """Count the D3 annotation shapes drawn on the heatmap subplot overlay."""
    return page.evaluate(
        """(plotId) => {
            const plot = document.getElementById(plotId);
            if (!plot) return -1;
            const overlay = plot.querySelector("[class*='-overplot-xy2']");
            if (!overlay) return 0;
            return overlay.querySelectorAll("polygon, path").length;
        }""",
        PLOT_ID,
    )


def _annotation_mark_count(page: Page) -> int:
    """Count committed annotation marks (bounding boxes + polygons) on the overlay."""
    return page.evaluate(
        """(plotId) => {
            const plot = document.getElementById(plotId);
            if (!plot) return -1;
            const overlay = plot.querySelector("[class*='-overplot-xy2']");
            if (!overlay) return 0;
            return overlay.querySelectorAll(
                "polygon[aria-label='polygon'], rect[aria-label='bounding-box']"
            ).length;
        }""",
        PLOT_ID,
    )


def _polygon_points(page: Page) -> str:
    return page.evaluate(
        """(plotId) => {
            const plot = document.getElementById(plotId);
            if (!plot) return null;
            const poly = plot.querySelector(
                "[class*='-overplot-xy2'] polygon[aria-label='polygon']"
            );
            return poly ? poly.getAttribute("points") : null;
        }""",
        PLOT_ID,
    )


def _select_tool(page: Page, tool: str, label: str) -> None:
    page.get_by_role("button", name="View Mode").click()
    page.locator("body").click()
    page.get_by_role("button", name=tool).click()
    page.get_by_test_id("select-annotation-label").click()
    page.get_by_test_id("popover").get_by_text(label).click()


def test_profile2d_threshold_toggle_preserves_manual_annotations(
    server_setup, page: Page
):
    """Toggling thresholding must not clear the user's own annotations."""
    setup_project(page)
    page.wait_for_timeout(1000)

    add_bounding_box(page)
    page.wait_for_timeout(500)
    manual = _annotation_mark_count(page)
    assert manual >= 1, "manual bounding box should be drawn"

    # Enabling thresholding adds polygons but must keep the manual annotation.
    page.get_by_role("button", name="Threshold").click()
    page.get_by_role("switch", name="Thresholding").click(force=True)
    page.wait_for_timeout(2500)
    assert _annotation_mark_count(page) > manual, (
        "enabling thresholding should keep the manual annotation and add polygons"
    )

    # Disabling removes the generated polygons but keeps the manual annotation.
    page.get_by_role("switch", name="Thresholding").click(force=True)
    page.wait_for_timeout(1500)
    assert _annotation_mark_count(page) >= manual, (
        "manual annotations must survive toggling thresholding off"
    )


def test_profile2d_polygon_finishes_on_ctrl_release(server_setup, page: Page):
    """Releasing Ctrl should commit an in-progress polygon so it stops tracking
    the cursor."""
    setup_project(page)
    page.wait_for_timeout(1000)

    _select_tool(page, "POLYGON", "NTM")
    box = page.get_by_label("profile-2d").bounding_box()
    assert box is not None
    # Three well-separated vertices within the heatmap (top 80% of the plot).
    vertices = [
        (box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.30),
        (box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.30),
        (box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.55),
    ]
    page.keyboard.down("Control")
    for vx, vy in vertices:
        page.mouse.move(vx, vy)
        page.mouse.down()
        page.mouse.up()
        page.wait_for_timeout(150)
    page.keyboard.up("Control")
    page.wait_for_timeout(600)

    committed = _polygon_points(page)
    assert committed is not None, "polygon should be committed after releasing Ctrl"

    # Moving the cursor afterwards must not reshape the finished polygon.
    page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.7)
    page.wait_for_timeout(400)
    page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
    page.wait_for_timeout(400)
    assert _polygon_points(page) == committed, (
        "a finished polygon must not follow the cursor"
    )


def test_profile2d_threshold_toggle_removes_annotations(server_setup, page: Page):
    """Toggling the threshold annotator off must remove its unsaved annotations."""
    setup_project(page)
    page.wait_for_timeout(1000)

    baseline = _annotation_shape_count(page)

    # Expand the Threshold panel and enable it - the annotator draws polygons.
    page.get_by_role("button", name="Threshold").click()
    page.get_by_role("switch", name="Thresholding").click(force=True)
    page.wait_for_timeout(2500)
    enabled = _annotation_shape_count(page)
    assert enabled > baseline, "enabling thresholding should draw polygon annotations"

    # Disabling must clear the generated (unsaved) polygons...
    page.get_by_role("switch", name="Thresholding").click(force=True)
    page.wait_for_timeout(1500)
    disabled = _annotation_shape_count(page)
    assert disabled <= baseline, "disabling thresholding should remove its annotations"

    # ...and they must not reappear after a zoom.
    cx, cy = _plot_centre(page)
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(800)
    assert _annotation_shape_count(page) <= baseline


def test_profile2d_saved_threshold_annotations_persist(server_setup, page: Page):
    """Saving threshold annotations marks them user-created, so toggling the
    annotator off no longer discards them."""
    project_id, sample_id, _reload = setup_project(page)
    page.wait_for_timeout(1000)

    page.get_by_role("button", name="Threshold").click()
    page.get_by_role("switch", name="Thresholding").click(force=True)
    page.wait_for_timeout(2500)
    enabled = _annotation_shape_count(page)
    assert enabled > 0, "enabling thresholding should draw polygon annotations"

    # Save, waiting for the PUT to the backend to complete.
    with page.expect_response(
        lambda r: (
            f"samples/{sample_id}/annotations" in r.url and r.request.method == "PUT"
        )
    ):
        page.get_by_role("button", name="Save").click(force=True)
    page.wait_for_timeout(1000)

    # Toggle the annotator off - the saved annotations must remain on the plot.
    page.get_by_role("switch", name="Thresholding").click(force=True)
    page.wait_for_timeout(1500)
    assert _annotation_shape_count(page) >= enabled, (
        "saved threshold annotations should persist after toggling off"
    )

    # And they should be stored as user-created + validated in the database.
    annotations = requests.get(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations"
    ).json()
    assert len(annotations) >= 1
    for annotation in annotations:
        assert annotation["created_by"] == "manual"
        assert annotation["validated"]


def test_profile2d_dragmode_stays_pan_while_drawing(server_setup, page: Page):
    """Problem 3: holding Ctrl to draw must not toggle the Plotly drag mode."""
    setup_project(page)
    page.wait_for_timeout(500)
    page.locator("body").click()

    assert get_dragmode(page) == "pan"

    # Holding Ctrl marks the app as "drawing"; the drag mode (and therefore the
    # Plotly toolbar's active button) must stay put rather than flipping to false.
    page.keyboard.down("Control")
    page.wait_for_timeout(300)
    dragmode_while_drawing = get_dragmode(page)
    page.keyboard.up("Control")

    assert dragmode_while_drawing == "pan"
