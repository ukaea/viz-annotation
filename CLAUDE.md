# TokTagger

Web platform for annotating tokamak diagnostic data (video / time-series / 2D-profile views) to build ML training sets. FastAPI backend in `toktagger/api`, Vite + React (TypeScript) frontend in `toktagger/ui`, MongoDB or an embedded `mongita` fallback for storage. Tests live at the repo root in `tests/`, not inside `toktagger/`.

For deeper context, read rather than guess: `docs/dev/API-Design.md` (endpoints + DB shape), `docs/creating_projects.md` (Project → Sample → Annotation → Model domain model), `docs/custom_dataloaders.md` / `docs/custom_models.md` (plugin registries).

## Commands

Backend (from repo root):
```sh
uv sync --all-extras   # or `uv sync` to skip the optional ML (ray/torch) extra
uv run ruff check toktagger tests --ignore=C901 && uv run ruff format --check toktagger tests
uv run --group dev pytest tests/api             # unit + API tests, models extra NOT installed
uv run --all-extras pytest tests/api            # same tests, models extra installed
uv run --group dev pytest tests/end_to_end      # Playwright e2e (run `playwright install --with-deps` once first)
```
Prefer running a single test (`pytest path/to/test_file.py -k test_name`) over the whole suite while iterating.

Frontend (from `toktagger/ui`):
```sh
npm ci
npx eslint . && npx prettier --check .
npx tsc --noEmit
npm run dev      # :5173 — expects the backend already running on :8002
npm run build    # writes into toktagger/api/static/, do not hand-edit that output
```

`pre-commit run --all-files` runs ruff + a check for stray `pdb` breakpoints; CI enforces both.

## Backend conventions

- IMPORTANT: `ray` and `torch` are the optional `pip install toktagger[models]` extra, never a base dependency. Guard anything importing them with `models_dependencies_installed()`. Keep `tests/conftest.py` and `tests/db_definitions.py` free of that import path — model/Ray-specific fixtures go in `tests/models_fixtures.py` / `tests/models_definitions.py` instead.
- Pydantic v2 idioms: `model_config = ConfigDict(...)`, not a nested `class Config`. Modern generics (`dict[str, Any]`, `X | None`), not `typing.Dict`/`Optional`. Imports at module top level, never inside a function.
- New config values go through a nested `pydantic.BaseModel` under `Settings` in `toktagger/api/config.py`, not ad hoc `os.environ` reads.
- Data loaders, ML models, and query strategies are added via their registry's `@Registry.register(...)` decorator (see `core/data_loaders.py`, `models/base.py`), not by special-casing a router.
- A new annotation/schema shape should reuse an existing base+subclass pattern (e.g. how `BoundingBox`/`VideoBoundingBox` relate) rather than being bespoke — look for the analogous type before adding a new one.
- New third-party dependencies need a real reason — don't add a second library that duplicates one already in use.

## Frontend conventions

- Prefer React Spectrum components and style props over Tailwind or custom CSS; reach for `UNSAFE_style` only once Spectrum's own props genuinely can't do it.
- No `any`. Use `unknown` with a real type guard, or the type a library (Annotorious, Plotly) already exports — don't invent a local type for a shape the library already provides.
- Domain types are Zod schemas with the TS type derived via `z.infer` (see `toktagger/ui/src/types.ts`); build variants with `BaseSchema.extend({...})` and `z.union([...])`, mirroring how the backend's Pydantic annotation schemas are structured. Parse with `Schema.safeParse()` once, check `.success`, then use `.data` — don't parse the same value twice or reach for a bare type assertion.
- When adding a new `useEffect`, check whether an existing one already has the same dependencies and purpose and extend that instead of adding a new adjacent effect. Don't merge or restructure existing effects as a drive-by while making an unrelated change — that's out of scope.
- View interaction state belongs in a React Context provider — not `localStorage`, custom DOM events, or globals. This is a deliberate architectural choice made after a prior implementation used those and became hard to maintain, not a style nit.
- Route all mutations to a third-party library's annotation store (Annotorious, Plotly) through a single function rather than scattering direct calls to its API; use D3 for custom interactive drawing/geometry layered on top instead of extending Plotly/Annotorious internals directly.

## Both: comments & typing

Default to zero comments; code should read clearly on its own. If something non-obvious needs a *why*, compress it into one single-line comment, never two. Multi-line blocks are a last resort, reserved for the rare case a single line genuinely cannot carry the context at all. A short docstring is fine when introducing a brand new function. Type everything; don't loosen a type or a test assertion just to make something pass without checking whether it's masking a real bug.

## E2E tests (`tests/end_to_end`)

- Wait for the specific network response an action triggers (`page.expect_response(lambda r: ...)`) instead of `page.wait_for_timeout(...)`.
- Locators should target ARIA roles / `data-testid`s. If an element is hard to locate, that's usually an accessibility gap in the component to fix, not a test problem to work around.
- Tests requiring the `models` extra are marked `@pytest.mark.models_enabled`; tests requiring it to be absent are marked `@pytest.mark.models_disabled`. An autouse fixture skips whichever doesn't match the current environment.

## Gotchas

- `toktagger/api/static/*` and `toktagger.example.toml` are CI-generated build output committed back to the branch by the `build` CI job — don't hand-edit them.
- CORS in `toktagger/api/main.py` only allows `http://localhost:5173`; the frontend dev server must run on that exact port for local API calls to work.
- PRs need sign-off from two other developers before merging (see `CONTRIBUTING.md`). Branch names are typically `<author-or-category>/<short-description>`, e.g. `wk9874/model_train_params`, `hotfix/arr_data_loader`.
