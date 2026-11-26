"""Adapter and isolated evaluation for the user_a profile bundle.

The bundled cases have their own profile-eval-v1 contract and intentionally do
not go through app.dataset.Case. Fake results are protocol checks; only the
marked real test measures model selection behavior.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest
from sqlalchemy import select

from app.agent import Agent
from app.config import settings
from app.db import products, traces, users
from app.memory.contracts import MemoryInput
from app.memory.recommendations import ceiling
from app.product_fixtures import import_product_fixtures, load_product_fixture
from app.rag import RAG


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "user_profile"
PROFILE_PATH = FIXTURE_DIR / "user_a_memory_profile.json"
CASES_PATH = FIXTURE_DIR / "profile_test_cases.json"


def load_profile_bundle():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    cases_doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if profile.get("schema_version") != "profile-seed-v1":
        raise ValueError("Unsupported user profile fixture version")
    if profile.get("actor") != {"tenant_id": "a", "user_id": "user_a"}:
        raise ValueError("Profile fixture must use the existing tenant-a user_a identity")
    if cases_doc.get("schema_version") != "profile-eval-v1":
        raise ValueError("Unsupported profile test-case version")
    cases = cases_doc.get("test_cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("Expected four user profile cases")
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate profile case_id")
    if any(case.get("actor") != "user_a" or case.get("mode") != "real" for case in cases):
        raise ValueError("Profile cases must target user_a and describe real-model expectations")
    if len(profile.get("memory_inputs", [])) != 4:
        raise ValueError("Expected four memory inputs")
    return profile, {case["case_id"]: case for case in cases}


@pytest.fixture
def profile_bundle():
    return load_profile_bundle()


@pytest.fixture
def profile_scenario(env, profile_bundle):
    profile, _ = profile_bundle
    assert env["a"].user_id == profile["actor"]["user_id"]
    assert env["a"].tenant_id == profile["actor"]["tenant_id"]

    first = import_product_fixtures(env["db"])
    second = import_product_fixtures(env["db"])
    assert first["total"] == 20 and second["inserted"] == 0
    with env["db"].tx() as connection:
        user = connection.execute(select(users).where(users.c.id == "user_a")).mappings().one()
        target = connection.execute(select(products).where(
            products.c.id == profile["gold_profile"]["matched_product_id"]
        )).mappings().one()
    assert user["tenant_id"] == "a"
    assert target["tenant_id"] == "a"
    return env


async def ensure_profile_memories(service, principal, profile):
    settings_row = service.update_settings(
        principal, auto_write_enabled=False, retrieval_enabled=True
    )
    assert settings_row["auto_write_enabled"] is False
    assert settings_row["retrieval_enabled"] is True

    expected = {row["subject_key"]: row for row in profile["memory_inputs"]}
    existing = {row.subject_key: row for row in await service.list_memories(principal=principal)}
    for subject_key, row in expected.items():
        current = existing.get(subject_key)
        if current is not None:
            assert current.memory_type == row["memory_type"]
            assert current.content == row["content"]
            continue
        await service.create_memory(
            principal=principal,
            data=MemoryInput.model_validate(row),
        )

    active = await service.list_memories(principal=principal)
    by_subject = {row.subject_key: row for row in active}
    assert set(by_subject) == set(expected)
    assert all(row.status == "active" for row in active)
    assert all(by_subject[key].content == value["content"] for key, value in expected.items())
    indexed = service.index.collection().get(
        ids=[row.memory_id for row in active], include=["metadatas"]
    )
    assert set(indexed["ids"]) == {row.memory_id for row in active}
    return active


def capture_model_context(agent, mode, monkeypatch):
    model = agent.harness.fake_model if mode == "deterministic" else agent.harness.model
    original = model.complete_with_tools
    captured = []

    async def wrapped(**kwargs):
        messages = kwargs["messages"]
        user_message = next((item for item in messages
                             if isinstance(item, dict) and item.get("role") == "user"), None)
        if user_message:
            payload = json.loads(user_message["content"])
            system_message = messages[0]
            captured.append({
                "payload": payload,
                "system": system_message.get("content", "") if isinstance(system_message, dict) else "",
            })
        return await original(**kwargs)

    monkeypatch.setattr(model, "complete_with_tools", wrapped)
    return captured


def trace_payload(env, result):
    with env["db"].tx() as connection:
        raw = connection.execute(select(traces.c.payload).where(traces.c.id == result["trace_id"])).scalar_one()
    return json.loads(raw)


def events(trace, kind):
    return [event for event in trace["events"] if event["type"] == kind]


def observed_tool_names(trace):
    return [event["name"] for event in events(trace, "loop.tool_observed")]


def selected_product_ids(trace):
    selected = events(trace, "loop.product_recommendation")
    return [product_id for event in selected for product_id in event.get("product_ids", [])]


def captured_memory(captures):
    return [entry for call in captures for entry in call["payload"].get("USER_MEMORY", [])]


def case_report(case_id, result, trace, captures, expected=None):
    search_events = events(trace, "memory.profile_loaded")
    injections = events(trace, "memory.context_injected")
    safe_answer = result["answer"]
    for memory in captured_memory(captures):
        safe_answer = safe_answer.replace(memory.get("content", ""), "[PROFILE_MEMORY_REDACTED]")
    report = {
        "case_id": case_id,
        "tool_names": observed_tool_names(trace),
        "tool_actions": [
            {
                "name": action["name"],
                "args": action["args"],
                "result_rows": [
                    {key: row.get(key) for key in (
                        "id", "name", "price_cents", "stock", "status", "carrier",
                        "tracking_no", "order_id", "eligible", "reason_code",
                    )}
                    for row in (action.get("result", {}).get("data") or [])
                    if isinstance(row, dict)
                ] if isinstance(action.get("result", {}).get("data"), list) else None,
            }
            for action in result["tool_actions"]
        ],
        "memory_matched_count": max((event.get("matched_count", 0) for event in search_events), default=0),
        "memory_context_ids": [ident for event in injections for ident in event.get("memory_ids", [])],
        "context_has_user_memory": bool(captured_memory(captures)),
        "selected_product_ids": selected_product_ids(trace),
        "recommendation_events": [
            {key: event.get(key) for key in ("product_ids", "source", "memory_ids")}
            for event in events(trace, "loop.product_recommendation")
        ],
        "reply_composed": [
            {key: event.get(key) for key in ("mode", "validation", "fact_count")}
            for event in events(trace, "loop.reply_composed")
        ],
        "actual_answer": safe_answer[:500],
        "answer_sha256": hashlib.sha256(result["answer"].encode("utf-8")).hexdigest(),
        "trace_id": result["trace_id"],
    }
    if expected is not None:
        report["expected"] = expected
    return report


def product_action(result):
    return next((action for action in result["tool_actions"] if action["name"] == "search_products"), None)


def product_actions(result):
    return [action for action in result["tool_actions"] if action["name"] == "search_products"]


def assert_no_private_memory_text(trace, profile):
    encoded = json.dumps(trace, ensure_ascii=False)
    for memory in profile["memory_inputs"]:
        assert memory["content"] not in encoded


def assert_profile_injected(captures, profile):
    contents = {row["content"] for row in profile["memory_inputs"]}
    assert captures
    for call in captures:
        assert {entry["content"] for entry in call["payload"].get("USER_MEMORY", [])} == contents
        assert "必须按当轮要求执行" in call["system"]
        assert all(content not in call["system"] for content in contents)


def full_profile_matches(rows, profile, max_price_cents):
    gold = profile["gold_profile"]
    count = 0
    for row in rows:
        attrs = json.loads(row["attributes_json"])
        if row["stock"] <= 0 or row["price_cents"] > max_price_cents:
            continue
        if attrs.get("form_factor") != gold["preferred_form_factor"]:
            continue
        if attrs.get("active_noise_cancellation") is not gold["need_active_noise_cancellation"]:
            continue
        if attrs.get("multipoint") is not gold["need_multipoint"]:
            continue
        if gold["prioritize_call_quality"] and attrs.get("call_noise_reduction") is not True:
            continue
        count += 1
    return count


def test_profile_bundle_contract_and_budget_parser(profile_bundle):
    profile, cases = profile_bundle
    assert set(cases) == {
        "profile-on-001", "profile-off-002", "profile-current-budget-003", "profile-unrelated-004"
    }
    assert ceiling(cases["profile-current-budget-003"]["user_message"]) == 20_000
    products_in_fixture = load_product_fixture()
    assert full_profile_matches(products_in_fixture, profile, profile["gold_profile"]["budget_max_cents"]) == 1
    assert full_profile_matches(products_in_fixture, profile, cases["profile-current-budget-003"]["gold"]["current_budget_max_cents"]) == cases["profile-current-budget-003"]["gold"]["full_requirements_candidate_count"]


@pytest.mark.asyncio
async def test_profile_fake_memory_context_live_products_budget_and_shipping(
    profile_scenario, profile_bundle, monkeypatch
):
    env = profile_scenario
    profile, cases = profile_bundle
    agent = env["agent"]
    service = agent.memory
    active = await ensure_profile_memories(service, env["a"], profile)
    second_load = await ensure_profile_memories(service, env["a"], profile)
    assert [row.memory_id for row in active] == [row.memory_id for row in second_load]
    memories_by_subject = {row.subject_key: row for row in active}
    captured = capture_model_context(agent, "deterministic", monkeypatch)

    on_case = cases["profile-on-001"]
    on_result = await agent.chat(env["a"], env["b"].conversation(env["a"])["id"],
                                on_case["user_message"], mode="deterministic")
    on_trace = trace_payload(env, on_result)
    on_report = case_report(on_case["case_id"], on_result, on_trace, captured, on_case["gold"])
    assert events(on_trace, "memory.profile_loaded")[0]["matched_count"] == len(profile["memory_inputs"])
    assert_profile_injected(captured, profile)
    assert [action["name"] for action in on_result["tool_actions"]].count("search_products") == 1
    assert set(on_case["gold"]["forbidden_tools"]).isdisjoint(observed_tool_names(on_trace))
    live = product_action(on_result)
    assert live and any(row["id"] == profile["gold_profile"]["matched_product_id"] for row in live["result"]["data"])
    assert_no_private_memory_text(on_trace, profile)

    captured.clear()
    off_case = cases["profile-off-002"]
    service.update_settings(env["a"], retrieval_enabled=off_case["setup"]["retrieval_enabled"])
    off_result = await agent.chat(env["a"], env["b"].conversation(env["a"])["id"],
                                 off_case["user_message"], mode="deterministic")
    off_trace = trace_payload(env, off_result)
    off_report = case_report(off_case["case_id"], off_result, off_trace, captured, off_case["gold"])
    assert not captured_memory(captured)
    assert not events(off_trace, "memory.context_injected")
    assert "search_products" in observed_tool_names(off_trace)
    assert all(action["name"] in off_case["gold"]["allowed_tools"] for action in off_result["tool_actions"])
    assert not re.search(r"已保存的偏好|长期记忆|你之前提到|根据你的画像", off_result["answer"])
    assert_no_private_memory_text(off_trace, profile)
    assert len(await service.list_memories(principal=env["a"])) == len(profile["memory_inputs"])
    service.update_settings(env["a"], retrieval_enabled=True)

    budget_case = cases["profile-current-budget-003"]
    captured.clear()
    budget_result = await agent.chat(env["a"], env["b"].conversation(env["a"])["id"],
                                    budget_case["user_message"], mode="deterministic")
    budget_trace = trace_payload(env, budget_result)
    budget_report = case_report(budget_case["case_id"], budget_result, budget_trace, captured, budget_case["gold"])
    assert_profile_injected(captured, profile)
    assert events(budget_trace, "memory.context_injected")
    assert "search_products" in observed_tool_names(budget_trace)
    assert full_profile_matches(load_product_fixture(), profile,
                                budget_case["gold"]["current_budget_max_cents"]) == 0
    budget_ids = selected_product_ids(budget_trace)
    if budget_ids:
        with env["db"].tx() as connection:
            item = connection.execute(select(products).where(products.c.id == budget_ids[0])).mappings().one()
        assert item["price_cents"] <= budget_case["gold"]["current_budget_max_cents"]
        assert item["stock"] > 0
        assert not re.search(r"(?:完全|全部)满足", budget_result["answer"])

    unrelated_case = cases["profile-unrelated-004"]
    captured.clear()
    unrelated_result = await agent.chat(env["a"], env["b"].conversation(env["a"])["id"],
                                       unrelated_case["user_message"], mode="deterministic")
    unrelated_trace = trace_payload(env, unrelated_result)
    unrelated_report = case_report(unrelated_case["case_id"], unrelated_result, unrelated_trace, captured, unrelated_case["gold"])
    assert_profile_injected(captured, profile)
    assert events(unrelated_trace, "memory.context_injected")
    assert any(action["name"] == "track_shipment" and
               action["args"].get("order_id") == unrelated_case["gold"]["order_id"]
               for action in unrelated_result["tool_actions"])
    assert set(unrelated_case["gold"]["forbidden_tools"]).isdisjoint(observed_tool_names(unrelated_trace))
    assert not re.search(r"预算|头戴式|耳机", unrelated_result["answer"])
    assert_no_private_memory_text(unrelated_trace, profile)

    comparison = {
        "mode": "fake_protocol_smoke_only",
        "memory_on_vs_off_tools_equal": on_report["tool_names"] == off_report["tool_names"],
        "memory_on_vs_off_selected_product_ids_equal": on_report["selected_product_ids"] == off_report["selected_product_ids"],
        "memory_on_vs_off_answers_differ": on_result["answer"] != off_result["answer"],
        "memory_on_has_context": on_report["context_has_user_memory"],
        "memory_off_has_context": off_report["context_has_user_memory"],
    }
    print("PROFILE_FAKE_REPORT=" + json.dumps({"cases": [on_report, off_report, budget_report, unrelated_report],
                                               "comparison": comparison}, ensure_ascii=False))


@pytest.mark.real
@pytest.mark.asyncio
async def test_profile_real_model_cases_record_trace_and_recommend_from_live_attributes(
    profile_scenario, profile_bundle, monkeypatch
):
    if not all((settings.llm_base_url, settings.llm_api_key, settings.llm_model)):
        pytest.skip("Explicit real LLM configuration required; no real-model result")
    if not Path(settings.embedding_model_path).is_dir():
        pytest.skip("Local BGE-M3 weights are required for the real profile retrieval check")

    env = profile_scenario
    profile, cases = profile_bundle
    real_cfg = env["cfg"].model_copy(update={
        "llm_base_url": settings.llm_base_url,
        "llm_api_key": settings.llm_api_key,
        "llm_model": settings.llm_model,
    })
    rag = RAG(env["db"], config=real_cfg)
    agent = Agent(env["b"], env["gateway"], real_cfg, rag)
    captured = capture_model_context(agent, "real", monkeypatch)
    failures = []
    reports = []

    async def run_case(case_id):
        case = cases[case_id]
        captured.clear()
        try:
            result = await agent.chat(env["a"], env["b"].conversation(env["a"])["id"],
                                     case["user_message"], mode="real")
            trace = trace_payload(env, result)
            report = case_report(case_id, result, trace, captured, case["gold"])
            reports.append(report)
            return case, result, trace, list(captured)
        except Exception as error:
            reports.append({"case_id": case_id, "error_type": type(error).__name__})
            failures.append(f"{case_id}: execution raised {type(error).__name__}")
            return case, None, None, []

    try:
        await ensure_profile_memories(agent.memory, env["a"], profile)

        on_case, on_result, on_trace, on_captures = await run_case("profile-on-001")
        if on_result:
            gold = profile["gold_profile"]
            matched = [event for event in events(on_trace, "memory.profile_loaded")
                       if event.get("matched_count") == len(profile["memory_inputs"])]
            injected_ids = {ident for event in events(on_trace, "memory.context_injected")
                            for ident in event.get("memory_ids", [])}
            capture_ids = {entry["memory_id"] for entry in captured_memory(on_captures)}
            if not matched or len(injected_ids) != len(profile["memory_inputs"]) or injected_ids != capture_ids:
                failures.append("profile-on-001: complete profile load and context injection was not observed")
            assert_profile_injected(on_captures, profile)
            names = observed_tool_names(on_trace)
            if "search_products" not in names or set(on_case["gold"]["forbidden_tools"]).intersection(names):
                failures.append("profile-on-001: product read or tool-permission check failed")
            live_rows = [row for action in product_actions(on_result)
                         for row in (action["result"]["data"] or [])]
            target = next((row for row in live_rows if row["id"] == gold["matched_product_id"]), None)
            if not target or target["price_cents"] != gold["matched_product_price_cents"] or target["stock"] <= 0:
                failures.append("profile-on-001: target was absent from live product results or facts were stale")
            selected = selected_product_ids(on_trace)
            if selected != [gold["matched_product_id"]] or gold["matched_product_name"] not in on_result["answer"]:
                failures.append("profile-on-001: model did not select the gold profile match as its single recommendation")
            mentioned = [row["name"] for row in load_product_fixture() if row["name"] in on_result["answer"]]
            if mentioned != [gold["matched_product_name"]]:
                failures.append("profile-on-001: final answer did not prioritize exactly one matching product")
            assert_no_private_memory_text(on_trace, profile)

        off_case = cases["profile-off-002"]
        agent.memory.update_settings(env["a"], retrieval_enabled=False)
        _, off_result, off_trace, off_captures = await run_case("profile-off-002")
        if off_result:
            if events(off_trace, "memory.context_injected") or any(
                "USER_MEMORY" in call["payload"] for call in off_captures
            ):
                failures.append("profile-off-002: profile memory was injected with retrieval disabled")
            if "search_products" not in observed_tool_names(off_trace):
                failures.append("profile-off-002: live product lookup was skipped")
            if re.search(r"已保存的偏好|长期记忆|你之前提到|根据你的画像", off_result["answer"]):
                failures.append("profile-off-002: answer claimed access to disabled profile memory")
            assert_no_private_memory_text(off_trace, profile)
        agent.memory.update_settings(env["a"], retrieval_enabled=True)

        budget_case = cases["profile-current-budget-003"]
        _, budget_result, budget_trace, budget_captures = await run_case("profile-current-budget-003")
        if budget_result:
            assert_profile_injected(budget_captures, profile)
            if "search_products" not in observed_tool_names(budget_trace):
                failures.append("profile-current-budget-003: live product lookup was skipped")
            selected = selected_product_ids(budget_trace)
            budget = budget_case["gold"]["current_budget_max_cents"]
            with env["db"].tx() as connection:
                rows = {row["id"]: dict(row) for row in connection.execute(select(products)).mappings()}
            if len(selected) != 1 or selected[0] not in rows or rows[selected[0]]["stock"] <= 0 or rows[selected[0]]["price_cents"] > budget:
                failures.append("profile-current-budget-003: selected product violated the current price or stock limit")
            if full_profile_matches(load_product_fixture(), profile, budget) != budget_case["gold"]["full_requirements_candidate_count"]:
                failures.append("profile-current-budget-003: static product fixture disagreed with its full-match gold")
            if re.search(r"(?:完全|全部)满足", budget_result["answer"]):
                failures.append("profile-current-budget-003: answer claimed the low-budget product meets all long-term requirements")
            if not any(row["name"] in budget_result["answer"] for row in load_product_fixture()
                       if row["price_cents"] <= budget and row["stock"] > 0):
                failures.append("profile-current-budget-003: no in-budget live product was recommended")
            assert_no_private_memory_text(budget_trace, profile)

        shipping_case = cases["profile-unrelated-004"]
        _, shipping_result, shipping_trace, shipping_captures = await run_case("profile-unrelated-004")
        if shipping_result:
            if not any(action["name"] == "track_shipment" and
                       action["args"].get("order_id") == shipping_case["gold"]["order_id"]
                       for action in shipping_result["tool_actions"]):
                failures.append("profile-unrelated-004: expected shipment lookup for O1002 was not made")
            assert_profile_injected(shipping_captures, profile)
            if set(shipping_case["gold"]["forbidden_tools"]).intersection(observed_tool_names(shipping_trace)):
                failures.append("profile-unrelated-004: unrelated product or refund tool was called")
            if re.search(r"预算|头戴式|耳机", shipping_result["answer"]):
                failures.append("profile-unrelated-004: answer mentioned unrelated headphone profile details")
            assert_no_private_memory_text(shipping_trace, profile)

        on_report = next((row for row in reports if row["case_id"] == "profile-on-001"), {})
        off_report = next((row for row in reports if row["case_id"] == "profile-off-002"), {})
        comparison = {
            "memory_on_vs_off_tools_equal": on_report.get("tool_names") == off_report.get("tool_names"),
            "memory_on_vs_off_selected_product_ids_equal": on_report.get("selected_product_ids") == off_report.get("selected_product_ids"),
            "memory_on_vs_off_answers_differ": on_report.get("answer_sha256") != off_report.get("answer_sha256"),
            "memory_on_context_ids": on_report.get("memory_context_ids", []),
            "memory_off_context_ids": off_report.get("memory_context_ids", []),
        }
        print("PROFILE_REAL_REPORT=" + json.dumps({"cases": reports, "comparison": comparison,
                                                   "failures": failures}, ensure_ascii=False))
        assert not failures, "Profile real-model evaluation failures: " + json.dumps(failures, ensure_ascii=False)
    finally:
        agent.memory.update_settings(env["a"], retrieval_enabled=True)
        await agent.close()
