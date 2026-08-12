# AGENTS.md

TokTagger is a web platform for annotating tokamak diagnostic data (video / time-series / 2D-profile views) to build ML training sets.

- `toktagger/api` — FastAPI backend (Pydantic schemas, MongoDB/`mongita` storage, Ray-based optional ML worker)
- `toktagger/ui` — Vite + React (TypeScript) frontend
- `tests/` — all tests, at repo root (not inside `toktagger/`); mirrors `api`/`ui` structure, plus `tests/end_to_end` for Playwright
- `docs/` — Zensical docs site; read `docs/dev/API-Design.md` (endpoints + DB shape) and `docs/creating_projects.md` (Project → Sample → Annotation → Model domain model) before making backend/API changes

## How to run

```sh
uv sync --all-extras          # backend deps; drop --all-extras to skip the optional ray/torch [models] extra
toktagger --reload --no-browser

npm --prefix toktagger/ui ci
npm --prefix toktagger/ui run dev   # :5173 — expects the backend already running on :8002
```
Or `docker compose -f docker-compose.dev.yml up` for the full stack (API + UI + Mongo + Mongo Express).

## Build, lint, test

```sh
uv run ruff check toktagger tests --ignore=C901 && uv run ruff format --check toktagger tests
npm --prefix toktagger/ui exec npx eslint . && npm --prefix toktagger/ui exec npx prettier --check .
npm --prefix toktagger/ui exec npx tsc --noEmit

uv run --group dev pytest tests/api             # unit + API tests, models extra NOT installed
uv run --all-extras pytest tests/api            # same tests, models extra installed
uv run --group dev pytest tests/end_to_end      # Playwright e2e; run `playwright install --with-deps` once first
npm --prefix toktagger/ui run build             # → toktagger/api/static/ (CI-generated, do not hand-edit)
```
Prefer running a single test (`pytest path/to/test_file.py -k test_name`) over a whole suite while iterating. `pre-commit run --all-files` runs ruff plus a stray-`pdb` check and is enforced in CI.

## Engineering conventions

**Backend**
- `ray`/`torch` are the optional `[models]` extra, never a base dependency. Guard imports with `models_dependencies_installed()`; keep `tests/conftest.py`/`tests/db_definitions.py` free of that import path — model-specific fixtures belong in `tests/models_fixtures.py`/`tests/models_definitions.py`.
- Pydantic v2 style: `model_config = ConfigDict(...)`, not nested `class Config`. Modern generics (`dict[str, Any]`, `X | None`). Imports at module top level, never inside a function.
- New config values extend `Settings` in `toktagger/api/config.py` as a nested `pydantic.BaseModel`, not ad hoc `os.environ` reads.
- Data loaders / models / query strategies register via their `@Registry.register(...)` decorator (`core/data_loaders.py`, `models/base.py`) rather than special-casing a router.
- A new annotation/schema shape should reuse an existing base+subclass pattern (e.g. how `BoundingBox`/`VideoBoundingBox` relate) rather than being bespoke — look for the analogous type before adding a new one.

**Frontend**
- React Spectrum components/style props over Tailwind or custom CSS; `UNSAFE_style` only once Spectrum props genuinely can't do it.
- No `any`. Use `unknown` with a real type guard, or the type a library (Annotorious, Plotly) already exports.
- Domain types are Zod schemas with the TS type derived via `z.infer` (`toktagger/ui/src/types.ts`); extend/union new variants the way existing ones are built. Parse with `Schema.safeParse()` once, check `.success`, then use `.data` — don't parse the same value twice or use a bare type assertion.
- When adding a new `useEffect`, check whether an existing one already has the same dependencies and purpose and extend that instead of adding a new adjacent effect. Don't merge or restructure existing effects as a drive-by while making an unrelated change — that's out of scope.
- View interaction state lives in a React Context provider — not `localStorage`, custom DOM events, or globals.
- Route third-party library (Annotorious, Plotly) mutations through one function; use D3 for custom drawing/geometry layered on top rather than extending those libraries directly.

**Both**: short one-line "why" comments, not multi-line comment blocks or docstrings. Don't loosen a type or a test assertion just to make something pass without first checking whether it's masking a real bug.

## PR expectations & constraints

- PRs need sign-off from two other developers before merging (`CONTRIBUTING.md`). Branch names are typically `<author-or-category>/<short-description>`, e.g. `wk9874/model_train_params`, `hotfix/arr_data_loader`.
- Don't hand-edit `toktagger/api/static/*` or `toktagger.example.toml` — both are CI-generated build output committed back to the branch.
- New third-party dependencies need a real reason; don't add a second library that duplicates one already in use.
- CORS in `toktagger/api/main.py` only allows `http://localhost:5173` — the frontend dev server must run on that exact port for local API calls to succeed.

## Definition of done

Ruff/ESLint/Prettier clean, `tsc --noEmit` clean, the relevant pytest suite(s) pass (include `tests/end_to_end` if UI behavior changed), and `docs/` is updated if the change affects a documented endpoint, config option, or UI workflow.

## Code Review Rules

### Optional ML dependency boundary
- Do not add `ray` or `torch` (or anything importing them unguarded) as a base dependency in `pyproject.toml`, or import them at module level outside a `models_dependencies_installed()` check.
  Safe path: guard the import, and put new model-specific test fixtures in `tests/models_fixtures.py`/`tests/models_definitions.py`, not `conftest.py`.

### Frontend state management
- Do not introduce `localStorage`, custom DOM `CustomEvent`s, or global/window variables to synchronize UI state across components.
  Safe path: use a React Context provider co-located with the view that owns the state.

### Frontend styling
- Do not add Tailwind classes or hand-written CSS to a component built from React Spectrum.
  Safe path: use Spectrum's layout/style props first; fall back to `UNSAFE_style` only if Spectrum genuinely can't express it.

### TypeScript typing
- Do not use `any`, and do not add a type assertion/cast to silence a type error.
  Safe path: use `unknown` with an explicit type guard, or the type the library already exports; fix the underlying mismatch instead of casting past it.

### Test integrity
- Do not loosen a test assertion (a wider exception type, a relaxed condition) purely to make a failing test pass.
  Safe path: confirm the original assertion isn't catching a real regression before relaxing it; ask if uncertain rather than loosening silently.
