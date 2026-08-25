"""
Runnable self-check for twtr-scraper's syndication block/retry path.

Run: python selfcheck_twtr_retry.py

Covers the distinction that makes a rate-limited account recoverable: a refused
request must raise _SyndicationBlocked so the account is retried, while a
genuinely empty timeline must return [] so it is not. Also asserts the account
loop parks blocked accounts for a later pass instead of dropping them for the
run, which is the failure mode this path is most prone to -- a 429 and an
account that never tweets used to produce the same "No recent tweets found".
"""
import io
import time
import urllib.error
import importlib.util

spec = importlib.util.spec_from_file_location("twtr_scraper", "twtr-scraper.py")
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _serving(handler):
    """Points the scraper's urlopen at `handler` for the duration."""
    original = tw.urllib.request.urlopen
    tw.urllib.request.urlopen = handler
    return original


def check_429_defers_and_arms_cooldown():
    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    tw._syndication_cooldown_until = 0.0
    original = _serving(refuse)
    try:
        try:
            tw._syndication_timeline("someacct")
        except tw._SyndicationBlocked:
            pass
        else:
            raise AssertionError("a 429 must raise _SyndicationBlocked, not return []")
    finally:
        tw.urllib.request.urlopen = original

    remaining = tw._syndication_cooldown_until - time.monotonic()
    assert remaining > 0, "a 429 must arm the shared cooldown"
    assert remaining <= tw.SYNDICATION_COOLDOWN_SECONDS + 1, remaining
    tw._syndication_cooldown_until = 0.0
    print("429 defers + arms cooldown: ok")


def check_challenge_page_defers():
    """A 200 carrying no __NEXT_DATA__ is a soft block, not an empty account."""
    original = _serving(lambda r, timeout=None: _FakeResponse(b"<html>nope</html>"))
    try:
        try:
            tw._syndication_timeline("someacct")
        except tw._SyndicationBlocked:
            pass
        else:
            raise AssertionError("a page with no timeline payload must raise _SyndicationBlocked")
    finally:
        tw.urllib.request.urlopen = original
    print("challenge page defers: ok")


def check_empty_timeline_is_not_a_block():
    """An account that genuinely has no tweets must not be retried forever."""
    body = (b'<script id="__NEXT_DATA__" type="application/json">'
            b'{"props":{"pageProps":{"timeline":{"entries":[]}}}}</script>')
    original = _serving(lambda r, timeout=None: _FakeResponse(body))
    try:
        assert tw._syndication_timeline("quietacct") == []
    finally:
        tw.urllib.request.urlopen = original
    print("empty timeline returns []: ok")


def check_cooldown_fails_fast_without_sleeping():
    """A live cooldown must skip syndication instantly, never sleep on it.

    This is the whole point of the cooldown: the run keeps moving on the other
    sources. A sleeping version stalls the entire account list behind one
    rate-limited endpoint.
    """
    tw._syndication_cooldown_until = time.monotonic() + 600.0

    def must_not_be_called(request, timeout=None):
        raise AssertionError("a cooled-down syndication must not issue a request")

    original = _serving(must_not_be_called)
    started = time.monotonic()
    try:
        try:
            tw._syndication_timeline("someacct")
        except tw._SyndicationBlocked:
            pass
        else:
            raise AssertionError("a live cooldown must raise _SyndicationBlocked")
    finally:
        tw.urllib.request.urlopen = original
        tw._syndication_cooldown_until = 0.0

    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"cooldown must fail fast, took {elapsed:.1f}s"
    print("cooldown fails fast without sleeping: ok")


def check_blocked_syndication_falls_through_to_other_sources():
    """A blocked best-source must not cost the account -- xtf/DDG still run."""
    tw._syndication_cooldown_until = time.monotonic() + 600.0
    calls = []

    def fake_xtf(username, limit):
        calls.append("xtf")
        return [(username, "111")]

    original_xtf = tw._fetch_timeline_refs
    tw._fetch_timeline_refs = fake_xtf
    try:
        refs, prefetched, blocked = tw.discover_account_tweet_refs("someacct", limit=5)
    finally:
        tw._fetch_timeline_refs = original_xtf
        tw._syndication_cooldown_until = 0.0

    assert calls == ["xtf"], "the fallback source must still be tried"
    assert refs == [("someacct", "111")], refs
    assert blocked is True, "the account must be flagged for a later syndication retry"
    print("blocked syndication falls through to other sources: ok")


def check_blocked_accounts_are_deferred_not_dropped():
    blocked = {"blockedone"}
    seen = []

    def fake_scrape(username, lsh, hash_by_id, **kwargs):
        seen.append(username)
        return None, username in blocked

    original_scrape = tw.scrape_account_tweets
    original_delay, original_blocked_delay = tw.ACCOUNT_DELAY_RANGE, tw.ACCOUNT_DELAY_BLOCKED_RANGE
    tw.scrape_account_tweets = fake_scrape
    tw.ACCOUNT_DELAY_RANGE = tw.ACCOUNT_DELAY_BLOCKED_RANGE = (0.0, 0.0)
    try:
        deferred = tw._scrape_account_list(["good", "blockedone", "alsogood"], None, None)
        assert deferred == ["blockedone"], deferred
        # The blocked account must not have cost the ones after it.
        assert seen == ["good", "blockedone", "alsogood"], seen

        blocked.clear()
        assert tw._scrape_account_list(deferred, None, None) == [], "retry pass must drain the deferral"
    finally:
        tw.scrape_account_tweets = original_scrape
        tw.ACCOUNT_DELAY_RANGE, tw.ACCOUNT_DELAY_BLOCKED_RANGE = original_delay, original_blocked_delay
    print("blocked accounts deferred then recovered: ok")


def check_one_broken_account_does_not_stop_the_list():
    """A genuine bug in one account must not defer it or end the run."""
    seen = []

    def fake_scrape(username, lsh, hash_by_id, **kwargs):
        seen.append(username)
        if username == "broken":
            raise ValueError("something genuinely broken")
        return None, False

    original_scrape = tw.scrape_account_tweets
    original_delay, original_blocked_delay = tw.ACCOUNT_DELAY_RANGE, tw.ACCOUNT_DELAY_BLOCKED_RANGE
    tw.scrape_account_tweets = fake_scrape
    tw.ACCOUNT_DELAY_RANGE = tw.ACCOUNT_DELAY_BLOCKED_RANGE = (0.0, 0.0)
    try:
        assert tw._scrape_account_list(["broken", "fine"], None, None) == []
        assert seen == ["broken", "fine"], seen
    finally:
        tw.scrape_account_tweets = original_scrape
        tw.ACCOUNT_DELAY_RANGE, tw.ACCOUNT_DELAY_BLOCKED_RANGE = original_delay, original_blocked_delay
    print("one broken account does not stop the list: ok")


if __name__ == "__main__":
    check_429_defers_and_arms_cooldown()
    check_challenge_page_defers()
    check_empty_timeline_is_not_a_block()
    check_cooldown_fails_fast_without_sleeping()
    check_blocked_syndication_falls_through_to_other_sources()
    check_blocked_accounts_are_deferred_not_dropped()
    check_one_broken_account_does_not_stop_the_list()
    print("all twtr retry checks passed")
