import json
import unittest
from unittest.mock import patch

import main


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeLLM:
    def __init__(self, response):
        self.response = response
        self.bound_tools = None

    def invoke(self, messages):
        return self.response

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self


class BraveSearchTests(unittest.TestCase):
    def test_needs_web_research_from_tool_hint(self):
        self.assertTrue(main._needs_web_research("Explain this library", "question", ["web_search"]))

    def test_needs_web_research_from_keyword_hint(self):
        self.assertTrue(main._needs_web_research("What is the latest OpenAI release?", "research", []))

    def test_needs_web_research_for_research_intent(self):
        self.assertTrue(main._needs_web_research("Can you do a search for ATM counts in Belgrade?", "research", []))

    def test_normalize_brave_results(self):
        payload = {
            "web": {
                "results": [
                    {
                        "title": "Brave Search API",
                        "url": "https://brave.com/search/api/",
                        "description": "Official API overview",
                        "age": "1 day ago",
                        "meta_url": {"hostname": "brave.com"},
                    }
                ]
            }
        }
        results = main._normalize_brave_results(payload)
        self.assertEqual(results[0]["title"], "Brave Search API")
        self.assertEqual(results[0]["source"], "brave.com")

    def test_parse_plain_text_tool_call(self):
        calls = main._parse_text_tool_calls('brave_web_search(query="atm count belgrade serbia", count=3)')
        self.assertEqual(calls[0]["name"], "brave_web_search")
        self.assertEqual(calls[0]["args"]["query"], "atm count belgrade serbia")
        self.assertEqual(calls[0]["args"]["count"], 3)

    def test_contains_plain_text_tool_call(self):
        self.assertTrue(main._contains_text_tool_call('brave_web_search(query="latest brave docs")'))

    def test_extract_context_override(self):
        text = '{"context_override": true, "mode": "web", "reason": "Need live web data", "query": "atm count belgrade serbia"}'
        override = main._extract_context_override(text)
        self.assertEqual(override["mode"], "web")
        self.assertEqual(override["query"], "atm count belgrade serbia")

    def test_after_respond_routes_context_override(self):
        state = {
            "messages": [main.AIMessage(content='{"context_override": true, "mode": "rag", "reason": "Need local docs", "query": "payment terminal error 57"}')]
        }
        self.assertEqual(main._after_respond(state), "dispatch_context_override")

    def test_route_context_override(self):
        self.assertEqual(main.route_context_override({"context_override_mode": "web"}), "web_research")
        self.assertEqual(main.route_context_override({"context_override_mode": "rag"}), "rag")

    def test_clarify_binds_tools(self):
        fake = _FakeLLM(main.AIMessage(content='brave_web_search(query="test")'))
        state = {
            "messages": [main.HumanMessage(content="help maybe?")],
            "confidence": 0.2,
        }
        with patch("main.get_llm", return_value=fake):
            result = main.clarify(state)
        self.assertEqual(result["messages"][0].content, 'brave_web_search(query="test")')
        self.assertIsNotNone(fake.bound_tools)

    def test_search_brave_raises_without_api_key(self):
        with patch.dict(main.SETTINGS, {"brave": {"api_key": ""}}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "API key"):
                main._search_brave("test query")

    def test_search_brave_returns_normalized_results(self):
        payload = {
            "web": {
                "results": [
                    {
                        "title": "Result One",
                        "url": "https://example.com/one",
                        "description": "Snippet one",
                        "meta_url": {"hostname": "example.com"},
                    }
                ]
            }
        }
        brave_settings = {
            "api_key": "test-key",
            "base_url": "https://api.search.brave.com/res/v1/web/search",
            "count": "5",
            "country": "us",
            "search_lang": "en",
            "safesearch": "moderate",
        }
        with patch.dict(main.SETTINGS, {"brave": brave_settings}, clear=False):
            with patch("main.urllib.request.urlopen", return_value=_FakeResponse(payload)):
                result = main._search_brave("result one")
        self.assertEqual(result["query"], "result one")
        self.assertEqual(result["results"][0]["url"], "https://example.com/one")


if __name__ == "__main__":
    unittest.main()
