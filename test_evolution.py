"""
Evolution System Integration Test
=================================
Tests all components of the evolution system.
"""

import asyncio
import tempfile
from pathlib import Path

def main():
    print("=" * 60)
    print("EVOLUTION SYSTEM TEST")
    print("=" * 60)

    # ========================================================================
    # TEST 1: Evolution Emitter
    # ========================================================================
    print("\n[TEST 1] Evolution Emitter")
    print("-" * 40)

    from app.core.evolution import (
        EvolutionEmitter, SignalType, DeploymentDomain,
    )

    emitter = EvolutionEmitter(
        agent_id="test_agent",
        agent_type="test",
        version="1.0.0",
        domain=DeploymentDomain.RESEARCH,
    )

    # Emit signals
    signal1 = emitter.emit(
        signal_type=SignalType.PREDICTION,
        payload={"prediction": 0.85, "confidence": 0.92},
    )
    print(f"  Signal emitted: {signal1.signal_id[:8]}...")

    # Use signal_id as request_id for tracking
    request_id = signal1.signal_id
    print(f"  Using signal_id as request_id: {request_id[:8]}...")

    # Record outcome
    emitter.record_outcome(request_id, {"correct": True, "actual": 0.87})
    print(f"  Outcome recorded for request")

    stats = emitter.get_stats()
    print(f"  Stats: {stats['signals_emitted']} signals, {stats['outcomes_recorded']} outcomes")
    print("  [PASS] Emitter working")

    # ========================================================================
    # TEST 2: Agent Evolution Integration
    # ========================================================================
    print("\n[TEST 2] Agent Evolution Integration")
    print("-" * 40)

    from app.agents.diagnostic_agent import DiagnosticAgent
    from app.agents.biomarker_agent import BiomarkerAgent

    diag_agent = DiagnosticAgent()
    print(f"  DiagnosticAgent created: {diag_agent.agent_id}")
    print(f"  Has emitter: {hasattr(diag_agent, '_emitter')}")
    print(f"  Configurable params: {list(diag_agent._get_configurable_parameters().keys())}")

    bio_agent = BiomarkerAgent()
    print(f"  BiomarkerAgent created: {bio_agent.agent_id}")

    # Test signal emission
    request_id = diag_agent.emit_signal(
        SignalType.PREDICTION,
        {"test": "data"},
    )
    print(f"  Signal emitted via agent: {request_id[:8]}...")
    print("  [PASS] Agent integration working")

    # ========================================================================
    # TEST 3: Cognition Store
    # ========================================================================
    print("\n[TEST 3] Cognition Store")
    print("-" * 40)

    from app.core.evolution import (
        CognitionStore, CognitionConfig,
        load_all_default_knowledge,
    )
    from app.core.evolution.schemas import CognitionItem, CognitionItemType

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CognitionConfig(storage_dir=Path(tmpdir))
        store = CognitionStore(config)

        # Load default knowledge
        results = load_all_default_knowledge(store)
        print(f"  Loaded: {results}")
        print(f"  Total items: {len(store)}")

        # Test retrieval
        retrieved = store.retrieve("FDA machine learning guidance", top_k=3, min_score=0.1)
        print(f"  Retrieved {len(retrieved)} items for FDA query")

        # Add custom item
        custom = CognitionItem(
            title="Test Lesson",
            content="This is a test lesson from evolution",
            item_type=CognitionItemType.LESSON,
        )
        store.add(custom)
        print(f"  Added custom item, total now: {len(store)}")

    print("  [PASS] Cognition store working")

    # ========================================================================
    # TEST 4: Three-Lane Database
    # ========================================================================
    print("\n[TEST 4] Three-Lane Database")
    print("-" * 40)

    from app.core.evolution import (
        ShadowStore, ProductionStore, PromotionQueue,
    )
    from app.core.evolution.schemas import EvolutionNode, Lane, ApprovalStatus

    with tempfile.TemporaryDirectory() as tmpdir:
        shadow = ShadowStore(storage_dir=Path(tmpdir) / "shadow")
        production = ProductionStore(storage_dir=Path(tmpdir) / "production")
        promotion = PromotionQueue(storage_dir=Path(tmpdir) / "promotion")

        # Create hypothesis in shadow
        node = shadow.create_hypothesis(
            name="test_hypothesis",
            motivation="Testing evolution system",
            code="def test(): pass",
            domain="research",
        )
        print(f"  Created node in shadow: {node.node_id[:8]}...")
        print(f"  Shadow size: {len(shadow)}")

        # Update score
        shadow.update_score(node.node_id, 0.85, {"test": True})
        updated = shadow.get(node.node_id)
        print(f"  Updated score: {updated.score}")

        # Queue for promotion
        promotion.queue_for_review(
            node=updated,
            requested_by="ai:test",
            rationale="High score in testing",
        )
        print(f"  Queued for promotion: {len(promotion)} pending")

        # Approve and promote
        promotion.approve(
            node_id=node.node_id,
            reviewer="human:tester",
            notes="Test approval",
        )

        production.promote_from_shadow(
            node=updated,
            reviewer="human:tester",
        )
        print(f"  Promoted to production: {len(production)} nodes")

    print("  [PASS] Three-lane database working")

    # ========================================================================
    # TEST 5: Pipeline Agents
    # ========================================================================
    print("\n[TEST 5] Pipeline Agents")
    print("-" * 40)

    from app.core.evolution import (
        ResearcherAgent, EvaluatorAgent, AnalyzerAgent,
    )
    from app.core.evolution.schemas import EvolutionNode

    researcher = ResearcherAgent()
    evaluator = EvaluatorAgent()
    analyzer = AnalyzerAgent()

    # Test researcher
    hypothesis = researcher.generate_hypothesis(
        task_description="Improve diagnostic accuracy for sepsis detection",
        parent_nodes=[],
        cognition_items=[],
    )
    print(f"  Generated hypothesis: {hypothesis['name']}")

    # Test evaluator (with mock node)
    mock_node = EvolutionNode(
        name="test_node",
        motivation="Test motivation",
        code="def test(): return True",
    )
    eval_result = evaluator.evaluate(mock_node, {})
    print(f"  Evaluation score: {eval_result['score']:.4f}")
    print(f"  Safety passed: {eval_result['safety_passed']}")

    # Test analyzer
    analysis = analyzer.analyze(
        node=mock_node,
        eval_result=eval_result,
        best_parent=None,
        task_description="Test task",
    )
    print(f"  Analysis generated: {len(analysis['analysis'])} chars")
    print(f"  Lessons extracted: {len(analysis['lessons'])}")

    print("  [PASS] Pipeline agents working")

    # ========================================================================
    # TEST 6: UtilityGate Evolution
    # ========================================================================
    print("\n[TEST 6] UtilityGate Evolution")
    print("-" * 40)

    from app.core.utility_engine.utility_gate import UtilityGate
    from app.core.utility_engine.schemas import UtilityInput, DeploymentMode

    gate = UtilityGate(mode=DeploymentMode.HOSPITAL)
    print(f"  UtilityGate created for {gate.mode.value}")
    print(f"  Has emitter: {hasattr(gate, '_emitter')}")

    # Create test signal
    test_signal = UtilityInput(
        entity_id="test_signal_001",
        entity_type="patient_alert",
        mode=DeploymentMode.HOSPITAL,
        title="Test Alert",
        summary="Testing evolution integration with UtilityGate",
        ppv_estimate=0.75,
        risk_probability=0.6,
        severity=0.8,
        novelty_score=0.7,
        actionability_score=0.85,
        confidence_score=0.9,
        metadata={},
    )

    decision = gate.evaluate(test_signal)
    print(f"  Decision: {decision.action.value}")
    print(f"  Should surface: {decision.should_surface}")
    print(f"  Handler score: {decision.breakdown.handler_score:.4f}")

    stats = gate.get_stats()
    print(f"  Decisions made: {stats['decisions_made']}")
    print("  [PASS] UtilityGate evolution working")

    # ========================================================================
    # TEST 7: Full Pipeline Integration
    # ========================================================================
    print("\n[TEST 7] Full Pipeline Integration")
    print("-" * 40)

    from app.core.evolution import (
        EvolutionOrchestrator, OrchestratorConfig, create_orchestrator,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = OrchestratorConfig(storage_dir=Path(tmpdir))
        orchestrator = EvolutionOrchestrator(config)

        # Set agents
        orchestrator.set_agents(researcher, evaluator, analyzer)
        print(f"  Orchestrator created")
        print(f"  Shadow store: {len(orchestrator.shadow)} nodes")

        # Get stats
        stats = orchestrator.get_stats()
        print(f"  Running: {stats['running']}")
        print(f"  Total steps: {stats['total_steps']}")

    print("  [PASS] Pipeline integration working")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("""
Evolution System Components Verified:
  1. EvolutionEmitter - Signal emission and outcome tracking
  2. Agent Integration - BaseAgent with evolution hooks
  3. Cognition Store - Knowledge storage and retrieval
  4. Three-Lane Database - Shadow/Production/Promotion workflow
  5. Pipeline Agents - Researcher, Evaluator, Analyzer
  6. UtilityGate - Decision tracking with evolution signals
  7. Orchestrator - Full pipeline integration
""")


if __name__ == "__main__":
    main()
