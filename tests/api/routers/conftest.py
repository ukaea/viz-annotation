"""Re-export the auth_setup fixture needed by test_auth.py (moved here from
tests/api/auth/).

It lives in tests/api/auth/conftest.py, whose scope doesn't extend to this
sibling directory. Importing it here makes pytest pick it up for tests in
this package too. Deliberately not importing that conftest's `_isolate_auth_cache`
(autouse) or `api_client` fixtures: autouse would apply to every test in this
directory, and `api_client` would shadow the root conftest's admin-authenticated
one that the other router tests rely on. auth_setup performs its own cache
isolation, so it doesn't need `_isolate_auth_cache` here.
"""

from tests.api.auth.conftest import auth_setup as auth_setup
