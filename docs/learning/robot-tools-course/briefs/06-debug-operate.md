# Module 6: 出问题时怎么查——从症状倒推层级

## Teaching Arc
- **Metaphor:** 飞机驾驶舱的分层仪表与黑匣子：先看哪块仪表变红，再去对应系统，不靠随机拆零件。
- **Opening hook:** status 是红叉、doctor 报 lock 不匹配、请求返回 503 或 500 时，第一步分别是什么？
- **Key insight:** status 查远端协议身份，doctor 查本机 runtime，typed exception 查单次调用，request_id 对应服务端日志，测试锁住已修复的坑。
- **Why should I care?:** 能结束“改一处、再坏一处”的 AI 调试循环，用证据缩小问题范围。

## Code Snippets (pre-extracted)

File: tests/test_launcher_status.py (lines 72-89)

    def test_status_rejects_swapped_service_ports(tmp_path, monkeypatch, capsys):
        path = _endpoints(
            tmp_path,
            "  fastfs: {host: 127.0.0.1, port: 5558}\n  sam3: {host: 127.0.0.1, port: 5556}\n",
        )
        _mock_health_by_port(
            monkeypatch,
            {
                5556: _health("fastfs"),
                5558: _health("sam3"),
            },
        )

        assert launcher.status(path) == 1

File: tests/test_launcher_parse.py (lines 62-79)

    def test_install_initializes_only_services_in_profile(tmp_path, monkeypatch):
        profile = _write(tmp_path, "services:\n  - name: sam3\n  - name: fastfs\n")
        initialized = []
        synced = []
        scripts = []
        monkeypatch.setattr(launcher, "_init_upstream", lambda spec: initialized.append(spec.service_id))
        monkeypatch.setattr(launcher, "_sync_one", lambda spec, extras: synced.append((spec.service_id, extras)))
        monkeypatch.setattr(
            launcher,
            "_run_runtime_script",
            lambda spec, script: scripts.append((spec.service_id, script)),
        )

        launcher.install(Path(profile))

        assert initialized == ["sam3", "fastfs"]
        assert synced == [("sam3", []), ("fastfs", [])]
        assert scripts == [("sam3", "install.sh"), ("fastfs", "install.sh")]

File: tests/test_graspgenx_client.py (lines 88-100)

    calls = []
    with GraspGenXClient(wait=False) as client:
        monkeypatch.setattr(client, "_request", lambda *args: calls.append(args))

        with pytest.raises((TypeError, ValueError)):
            _call_inference(
                client,
                method_name,
                grasp_threshold=grasp_threshold,
                topk_num_grasps=topk_num_grasps,
            )

    assert calls == []

## Interactive Elements
- [x] **Code↔English translation** — three regression tests as executable promises.
- [x] **Quiz** — 5 symptom-to-first-check scenarios: connect error, protocol mismatch, busy, real marker mismatch, inference 500.
- [ ] **Group chat animation**
- [x] **Data flow animation** — troubleshooting decision tree from command/output to layer and next evidence.
- [ ] **Drag-and-drop**
- [x] **Other** — spot-the-bug challenge using malformed health metadata or boolean selection; error card deck; final “I can add a service” mastery checklist.

## Reference Files to Read
- interactive-elements.md → Code ↔ English Translation Blocks; Spot the Bug Challenge; Scenario Quiz; Flow Diagrams; Pattern/Feature Cards; Callout Boxes; Glossary Tooltips
- design-system.md → Animations & Transitions; Module Structure; Responsive Breakpoints
- content-philosophy.md → all content rules
- gotchas.md → full checklist

## Connections
- **Previous module:** newly added service needs operational confidence.
- **Next module:** none; close by connecting all six modules into a repeatable workflow.
- **Tone/style notes:** Encourage evidence-first debugging. Define traceback, request ID, compatibility, timeout, admission/capacity, fixture, mock/monkeypatch and regression test.
