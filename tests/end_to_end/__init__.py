try:
    from playwright.sync_api import Page, expect
except ImportError:
    Page = None  # type: ignore[assignment,misc]
    expect = None  # type: ignore[assignment]


def form_check(page: Page, submit_button_name):
    expect(page.get_by_text("Model Parameters")).to_be_visible()
    # Check form has correct number, string, boolean, literal input types
    expect(page.get_by_role("textbox", name="Final Score")).to_be_visible()
    expect(page.get_by_role("button", name="Increase Final Score")).to_be_visible()
    expect(page.get_by_role("button", name="Decrease Final Score")).to_be_visible()
    expect(page.get_by_role("textbox", name="Test String")).to_be_visible()
    expect(page.get_by_role("checkbox", name="Test Bool")).to_be_visible()
    expect(page.get_by_role("combobox", name="Test Selection")).to_be_visible()

    # Try entering text into numberbox, shouldn't be allowed
    page.get_by_role("textbox", name="Final Score").fill("Test Text in Number Field")
    expect(page.get_by_text("Test Text in Number Field")).to_be_hidden()

    # Try entering number > 100, shouldnt be allowed, clamp to highest allowed val
    page.get_by_role("textbox", name="Final Score").fill("200")
    page.get_by_role("textbox", name="Final Score").press("Enter")
    expect(page.get_by_role("textbox", name="Final Score")).to_have_value("100")

    # Try entering number < 50, shouldnt be allowed, clamp to lowest allowed val
    page.get_by_role("textbox", name="Final Score").fill("10")
    page.get_by_role("textbox", name="Final Score").press("Enter")
    expect(page.get_by_role("textbox", name="Final Score")).to_have_value("50")

    # Try pressing up/down arrow, should increment/decrement by 1 since it is an int
    page.get_by_role("button", name="Increase Final Score").click()
    expect(page.get_by_role("textbox", name="Final Score")).to_have_value("51")
    page.get_by_role("button", name="Decrease Final Score").click()
    expect(page.get_by_role("textbox", name="Final Score")).to_have_value("50")

    # Checkbox should be ticked by default
    expect(page.get_by_role("checkbox", name="Test Bool")).to_be_checked()

    # Combobox should have two possible options
    page.get_by_role("button", name="Test Selection").scroll_into_view_if_needed()
    page.get_by_role("button", name="Test Selection").click()
    expect(page.get_by_role("option", name="selection_1", exact=True)).to_be_visible()
    expect(page.get_by_role("option", name="selection_2", exact=True)).to_be_visible()
    page.get_by_role("option", name="selection_1").click()

    # Try pressing submit
    page.get_by_role("button", name=submit_button_name, exact=True).click()

    # Should give validation error for missing string
    expect(page.get_by_text("There are problems with your submission:")).to_be_visible()

    page.get_by_role("textbox", name="Test String").fill("Entered")
